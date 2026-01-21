/* 
TMA to shared memory (bf16)
0.6 tflops lol (but that's ok it's just for testing tma)

Layout convention:
  - A is NxK, row-major:     A[n,k] = A_storage[n * K + k]
  - B is KxN, column-major:  B[k,n] = B_storage[n * K + k]  (stored as NxK)
  - C is NxN, row-major:     C[n,m] = C_storage[n * N + m]

This means B is stored transposed compared to standard row-major GEMM.
*/

#include "boilerplate.cuh"
#include "helpers.cuh"

using bf16 = __nv_bfloat16;

#define NUM_THREADS 32
#define TILE_SIZE 32

__global__ void kernel(bf16 *A, bf16 *B, bf16 *C, int N, 
                        const __grid_constant__ CUtensorMap A_tmap, const __grid_constant__ CUtensorMap B_tmap) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;
    int tile_dim = blockDim.x;
    int num_tiles = N / tile_dim;

    __shared__ alignas(128) bf16 A_tile[TILE_SIZE * TILE_SIZE];
    __shared__ alignas(128) bf16 B_tile[TILE_SIZE * TILE_SIZE];
    int A_smem = static_cast<int>(__cvta_generic_to_shared(A_tile));
    int B_smem = static_cast<int>(__cvta_generic_to_shared(B_tile));

    float acc = 0.0f;

    // create shared barrier bar
    __shared__ uint64_t mbar[1];
    const int mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbar));
    if (is_elected()) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(1));
        asm volatile("fence.mbarrier_init.release.cluster;");  // visible to async proxy
    }
    __syncthreads();
    int phase = 0;

    for (int t = 0; t < num_tiles; t++) {
        // A_tile: load tile at (col=t*TILE_SIZE, row=tile_row*TILE_SIZE) from row-major A
        // B_tile: load tile at (col=t*TILE_SIZE, row=tile_col*TILE_SIZE) from col-major B (stored as NxK)
        if (is_elected()) {
            int off_k = t * tile_dim;
            int off_m = tile_row * tile_dim;
            int off_n = tile_col * tile_dim;
            tma_2d_gmem2smem(A_smem, &A_tmap, off_k, off_m, mbar_addr);
            tma_2d_gmem2smem(B_smem, &B_tmap, off_k, off_n, mbar_addr);
            constexpr int copy_size= 2 * TILE_SIZE * TILE_SIZE * sizeof(bf16);
            // not a race condition that we call expect bytes after the tma_2d_gmem2smem
            // since phase ends when tx_count reaches 0, tx_count can go negative with the tma completion first
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(mbar_addr), "r"(copy_size) : "memory");
        }
        
        mbarrier_wait(mbar_addr, phase);
        phase ^= 1;

        for (int k = 0; k < tile_dim; k++) {
            acc += __bfloat162float(A_tile[y * tile_dim + k]) * __bfloat162float(B_tile[x * tile_dim + k]);
        }

        __syncthreads();
    }

    C[(tile_row * tile_dim + y) * N + tile_col * tile_dim + x] = __float2bfloat16(acc);
}

int main() {
    // C = A @ B where B is stored column-major (transposed in memory)
    // A is NxN row-major, B is NxN col-major (stored as NxN), C is NxN row-major
    int N = 2048; 

    // Allocate device memory
    bf16 *A_d, *B_d, *C_d, *C_ref;
    CUDACHECK(cudaMalloc(&A_d, sizeof(bf16) * N * N));
    CUDACHECK(cudaMalloc(&B_d, sizeof(bf16) * N * N));
    CUDACHECK(cudaMalloc(&C_d, sizeof(bf16) * N * N));
    CUDACHECK(cudaMalloc(&C_ref, sizeof(bf16) * N * N));

    // Initialize matrices on device
    fill_random(A_d, N * N, 2024);
    fill_random(B_d, N * N, 2025);
    CUDACHECK(cudaDeviceSynchronize());

    // Compute reference GEMM on device (B is column-major)
    reference_gemm_Bcolmaj(C_ref, A_d, B_d, N);
    CUDACHECK(cudaDeviceSynchronize());

    dim3 grid(N / NUM_THREADS, N / NUM_THREADS);
    dim3 block(NUM_THREADS, NUM_THREADS);

    CUtensorMap A_tmap{};
    CUtensorMap B_tmap{};
    CUresult res1 = init_tmap_2d(&A_tmap, A_d, N, N, TILE_SIZE, TILE_SIZE, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
    CUresult res2 = init_tmap_2d(&B_tmap, B_d, N, N, TILE_SIZE, TILE_SIZE, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);

    // Warmup iterations to stabilize GPU state (clocks, cache, etc.)
    const int num_warmups = 100;
    for (int i = 0; i < num_warmups; i++) {
        kernel<<<grid, block>>>(A_d, B_d, C_d, N, A_tmap, B_tmap);
    }
    CUDACHECK(cudaDeviceSynchronize());

    // Benchmark with multiple iterations for averaging
    const int num_iters = 100;
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        kernel<<<grid, block>>>(A_d, B_d, C_d, N, A_tmap, B_tmap);
    }
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    // Duration and TFLOPs (averaged over num_iters)
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double microseconds = milliseconds * 1000.0 / num_iters;  // Average time per iteration
    double flops = double(2.0) * N * N * N;
    double tflops = (flops / microseconds) / 1e6;
    std::cout << "Average kernel execution time: " << microseconds << " us\n";
    std::cout << "Performance: " << tflops << " TFLOPs\n";

    // Check correctness (takes device pointers)
    check_correctness(C_d, C_ref, N * N);

    CUDACHECK(cudaFree(A_d));
    CUDACHECK(cudaFree(B_d));
    CUDACHECK(cudaFree(C_d));
    CUDACHECK(cudaFree(C_ref));
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));

    return 0;
}
