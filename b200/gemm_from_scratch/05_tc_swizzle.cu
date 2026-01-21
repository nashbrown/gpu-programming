/* 
added shared mem swizzling to 04
to reduce bank conflicts... but probably more so to get a 128byte width tile instead of 16byte
420 tflops

Layout convention:
  - A is M×K, row-major:     A[m,k] = A_storage[m * K + k]
  - B is K×N, column-major:  B[k,n] = B_storage[n * K + k]  (stored as N×K)
  - C is M×N, row-major:     C[m,n] = C_storage[m * N + n]
*/

#include "boilerplate.cuh"
#include "helpers.cuh"

using bf16 = __nv_bfloat16;

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;
constexpr int THREAD_BLOCK_SIZE = WARP_SIZE * NUM_WARPS;

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 256;
constexpr int BLOCK_K = 256;
constexpr uint32_t CORE_MATRIX_WIDTH_BYTES = 128; // swizzle bytes changes width
constexpr uint32_t CORE_MATRIX_WIDTH = CORE_MATRIX_WIDTH_BYTES / sizeof(bf16);
constexpr int MMA_K = 16;  // bf16: 16 elements per MMA K-step (= 32 bytes)

// Instruction descriptor for tcgen05.mma
constexpr uint32_t IDESC = 
      (1U << 4U)                         // bits 4-6:   dtype = FP32 (accumulator)
    | (1U << 7U)                         // bits 7-9:   atype = BF16
    | (1U << 10U)                        // bits 10-12: btype = BF16
    | ((BLOCK_N / 8) << 17U)             // bits 17-23: MMA_N = 256/8 = 32
    | ((BLOCK_M / 16) << 24U);           // bits 24-30: MMA_M = 128/16 = 8


__global__ __launch_bounds__(THREAD_BLOCK_SIZE)
void kernel(bf16 *A, bf16 *B, bf16 *C, int M, int N, int K,
            const __grid_constant__ CUtensorMap A_tmap, const __grid_constant__ CUtensorMap B_tmap) {
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    int tile_m = blockIdx.y;  // which M-tile
    int tile_n = blockIdx.x;  // which N-tile
    int num_k_tiles = K / BLOCK_K;

    // Shared memory for A tile (BLOCK_M × BLOCK_K) and B tile (BLOCK_N × BLOCK_K)
    __shared__ alignas(1024) bf16 A_tile[BLOCK_M * BLOCK_K];
    __shared__ alignas(1024) bf16 B_tile[BLOCK_N * BLOCK_K];
    const int A_smem = static_cast<int>(__cvta_generic_to_shared(A_tile));
    const int B_smem = static_cast<int>(__cvta_generic_to_shared(B_tile));

    __shared__ uint64_t mbar[1]; // mbarrier
    const int mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbar));
    __shared__ int tmem_addr[1]; // tmem address, returned by tcgen05.alloc

    // Setup: initialize mbarrier (warp 0, 1 thread) and allocate tmem (warp 1, all 32 threads)
    if (warp_id == 0 && is_elect_sync()) {
        // Only 1 thread initializes the mbarrier
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(1));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    else if (warp_id == 1) {
        // All 32 threads in warp 1 must call tcgen05.alloc together
        // It allocates BLOCK_N columns of tensor memory (128 rows are always allocated)
        const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" 
            :: "r"(addr), "r"(BLOCK_N));
    }
    __syncthreads();  // Make mbarrier init and tmem allocation visible to all threads

    // Read the tensor memory address
    const int taddr = tmem_addr[0];
    int phase = 0;

    for (int t = 0; t < num_k_tiles; t++) {
        
        if (warp_id == 0 && is_elect_sync()) {
            int off_m = tile_m * BLOCK_M;
            int off_n = tile_n * BLOCK_N;

            for (int k = 0; k < BLOCK_K / CORE_MATRIX_WIDTH; k++) {
                int off_k = t * BLOCK_K + k * CORE_MATRIX_WIDTH;
                tma_2d_gmem2smem(A_smem + k * BLOCK_M * CORE_MATRIX_WIDTH_BYTES, &A_tmap, off_k, off_m, mbar_addr);
                tma_2d_gmem2smem(B_smem + k * BLOCK_N * CORE_MATRIX_WIDTH_BYTES, &B_tmap, off_k, off_n, mbar_addr);
            }
            
            constexpr int copy_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(bf16);
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(mbar_addr), "r"(copy_size) : "memory");
        }
        // tma completes when net tx bytes reaches 0
        mbarrier_wait(mbar_addr, phase);
        phase ^= 1;

        // Fence required after thread sync, before MMA
        asm volatile("tcgen05.fence::after_thread_sync;");

        // Issue MMA operations (only 1 thread)
        if (warp_id == 0 && is_elect_sync()) {
            // Manually unroll 1st iteration to disable accumulation
            uint64_t a_desc = make_smem_desc_swizzled(A_smem);
            uint64_t b_desc = make_smem_desc_swizzled(B_smem);
            tcgen05_mma_f16(taddr, a_desc, b_desc, IDESC, t);
            
            for (int k2 = 1; k2 < CORE_MATRIX_WIDTH / MMA_K; k2++) {
                int a_addr = A_smem + k2 * MMA_K * 2;
                int b_addr = B_smem + k2 * MMA_K * 2;
                uint64_t a_desc = make_smem_desc_swizzled(a_addr);
                uint64_t b_desc = make_smem_desc_swizzled(b_addr);
                tcgen05_mma_f16(taddr, a_desc, b_desc, IDESC, 1);
            }
            // k1 selects the (BLOCK_M, 128B) tile.
            // k2 selects the (BLOCK_M, 32B) tile, whose rows are swizzled.
            for (int k1 = 1; k1 < BLOCK_K / CORE_MATRIX_WIDTH; k1++) {
                for (int k2 = 0; k2 < CORE_MATRIX_WIDTH / MMA_K; k2++) {
                    int a_addr = A_smem + k1 * CORE_MATRIX_WIDTH_BYTES * BLOCK_M + k2 * MMA_K * 2;
                    int b_addr = B_smem + k1 * CORE_MATRIX_WIDTH_BYTES * BLOCK_N + k2 * MMA_K * 2;
                    uint64_t a_desc = make_smem_desc_swizzled(a_addr);
                    uint64_t b_desc = make_smem_desc_swizzled(b_addr);
                    tcgen05_mma_f16(taddr, a_desc, b_desc, IDESC, 1);
                }
            }
            
            // Commit MMA operations and signal mbarrier when MMA completes
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                :: "r"(mbar_addr) : "memory");
        }

        // Wait for MMA to complete (smem can be overwritten now)
        mbarrier_wait(mbar_addr, phase);
        phase ^= 1;
    }

    // Fence required before tcgen05.ld, after tcgen05.mma
    asm volatile("tcgen05.fence::after_thread_sync;");

    // Output tile offsets in global memory
    int off_m = tile_m * BLOCK_M;
    int off_n = tile_n * BLOCK_N;

    // Load 8 columns from tmem at a time (BLOCK_N / 8 = 32 iterations)
    // Layout D: each warp handles 32 rows of the 128-row output
    for (int n = 0; n < BLOCK_N / 8; n++) {
        // tmem address: upper 16 bits = row, lower 16 bits = column
        int row = warp_id * 32;      // Warp 0→0, 1→32, 2→64, 3→96
        int col = n * 8;
        int addr = taddr + (row << 16) + col;

        // Load 8 FP32 values from tensor memory
        float tmp[8];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
              "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
            : "r"(addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;");

        // Convert FP32 to BF16 (pack pairs into bfloat162)
        nv_bfloat162 out[4];
        for (int i = 0; i < 4; i++) {
            out[i] = __float22bfloat162_rn({tmp[i * 2], tmp[i * 2 + 1]});
        }

        // Write 16 bytes (8 bf16) to global memory C
        bf16 *out_ptr = C + (off_m + tid) * N + (off_n + n * 8);
        reinterpret_cast<int4 *>(out_ptr)[0] = reinterpret_cast<int4 *>(out)[0];
    }

    // ensure threads finish reading from tmem before deallocation
    __syncthreads();
    if (warp_id == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" 
            :: "r"(taddr), "r"(BLOCK_N));
    }
}

int main() {
    // C = A @ B where B is stored column-major (transposed in memory)
    // A is M×K row-major, B is K×N col-major (stored as N×K), C is M×N row-major
    int M = 2048, N = 2048, K = 2048;

    // Allocate device memory
    bf16 *A_d, *B_d, *C_d, *C_ref;
    CUDACHECK(cudaMalloc(&A_d, sizeof(bf16) * M * K));
    CUDACHECK(cudaMalloc(&B_d, sizeof(bf16) * N * K));
    CUDACHECK(cudaMalloc(&C_d, sizeof(bf16) * M * N));
    CUDACHECK(cudaMalloc(&C_ref, sizeof(bf16) * M * N));

    // Initialize matrices on device
    fill_random(A_d, M * K, 2024);
    fill_random(B_d, N * K, 2025);
    CUDACHECK(cudaDeviceSynchronize());

    // Compute reference GEMM on device (B is column-major)
    reference_gemm_Bcolmaj(C_ref, A_d, B_d, M, N, K);
    CUDACHECK(cudaDeviceSynchronize());

    // Grid: one block per output tile
    dim3 grid(N / BLOCK_N, M / BLOCK_M);
    dim3 block(THREAD_BLOCK_SIZE);

    // Create tensor maps
    // A is M×K (row-major): width=K, height=M
    // B is stored as N×K (col-major view of K×N): width=K, height=N
    CUtensorMap A_tmap{};
    CUtensorMap B_tmap{};
    CUresult res1 = init_tmap_2d(&A_tmap, A_d, K, M, CORE_MATRIX_WIDTH, BLOCK_M, CU_TENSOR_MAP_SWIZZLE_128B);
    CUresult res2 = init_tmap_2d(&B_tmap, B_d, K, N, CORE_MATRIX_WIDTH, BLOCK_N, CU_TENSOR_MAP_SWIZZLE_128B);

    // Warmup iterations to stabilize GPU state (clocks, cache, etc.)
    const int num_warmups = 100;
    for (int i = 0; i < num_warmups; i++) {
        kernel<<<grid, block>>>(A_d, B_d, C_d, M, N, K, A_tmap, B_tmap);
    }
    CUDACHECK(cudaDeviceSynchronize());

    // Benchmark with multiple iterations for averaging
    const int num_iters = 100;
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        kernel<<<grid, block>>>(A_d, B_d, C_d, M, N, K, A_tmap, B_tmap);
    }
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    // Duration and TFLOPs (averaged over num_iters)
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double microseconds = milliseconds * 1000.0 / num_iters;  // Average time per iteration
    double flops = double(2.0) * M * N * K;
    double tflops = (flops / microseconds) / 1e6;
    std::cout << "Average kernel execution time: " << microseconds << " us\n";
    std::cout << "Performance: " << tflops << " TFLOPs\n";

    // Check correctness (takes device pointers)
    check_correctness(C_d, C_ref, M * N);

    CUDACHECK(cudaFree(A_d));
    CUDACHECK(cudaFree(B_d));
    CUDACHECK(cudaFree(C_d));
    CUDACHECK(cudaFree(C_ref));
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));

    return 0;
}
