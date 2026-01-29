/* 
persistent kernel with static scheduling

Layout convention:
  - A is M×K, row-major:     A[m,k] = A_storage[m * K + k]
  - B is K×N, column-major:  B[k,n] = B_storage[n * K + k]  (stored as N×K)
  - C is M×N, row-major:     C[m,n] = C_storage[m * N + n]


OUTLINE

tma warp:
needs to wait on mma warp to consume from shared memory, signal mma warp when it's done

mma warp:
before loop, wait on tmem_empty
{in its loop over k tiles:
needs to wait on tma warp to load to shared memory, signal}
signal tmem_full at the end

epilogue warps:
wait on tmem_full 
do work
signal tmem_empty 
*/

#include "boilerplate.cuh"
#include "helpers.cuh"

using bf16 = __nv_bfloat16;

#include <thread>
#include <chrono>

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 6;  // 1 tma, 1 mma, 4 epilogue
constexpr int THREAD_BLOCK_SIZE = WARP_SIZE * NUM_WARPS;
constexpr int NUM_SM = 148;
constexpr int TB_SWIZZLE_N = 4;
constexpr int CLUSTER_SIZE = 2;
constexpr int STAGES = 7;

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 256;
constexpr int CLUSTER_N = BLOCK_N / CLUSTER_SIZE;
constexpr int BLOCK_K = 64;
constexpr uint32_t CORE_MATRIX_WIDTH_BYTES = 128; // swizzle bytes change width
constexpr uint32_t CORE_MATRIX_WIDTH = CORE_MATRIX_WIDTH_BYTES / sizeof(bf16);
constexpr int MMA_K = 16;  // bf16: 16 elements per MMA K-step (= 32 bytes)

// Instruction descriptor for tcgen05.mma
constexpr int MMA_M = BLOCK_M * CLUSTER_SIZE;  // 128 for 1 SM, 256 for 2 SMs
constexpr uint32_t IDESC = 
      (1U << 4U)                         // bits 4-6:   dtype = FP32 (accumulator)
    | (1U << 7U)                         // bits 7-9:   atype = BF16
    | (1U << 10U)                        // bits 10-12: btype = BF16
    | ((BLOCK_N / 8) << 17U)             // bits 17-23: MMA_N = 256/8 = 32
    | ((MMA_M / 16) << 24U);             // bits 24-30: MMA_M = 256/16 = 16


// non block swizzling version
// __device__ __forceinline__ int2 compute_position(int bid, int grid_n) {
//     // pair cta's must have adjacent tile_m and same tile_n
//     int2 tile;
//     tile.y = bid / (grid_n * CLUSTER_SIZE) * CLUSTER_SIZE + (bid % CLUSTER_SIZE);
//     tile.x = (bid / CLUSTER_SIZE) % grid_n;
//     return tile;
// }

// thread block swizzling in bands to for better L2 cache reuse
__device__ __forceinline__ int2 compute_position(int bid, int cta_rank, int grid_m, int grid_n) {
    const int cluster_id = bid / CLUSTER_SIZE;
    const int clusters_m = grid_m / CLUSTER_SIZE;
    const int tiles_per_band = clusters_m * TB_SWIZZLE_N;

    const int band = cluster_id / tiles_per_band;
    const int in_band = cluster_id - band * tiles_per_band;

    const int cluster_m = in_band / TB_SWIZZLE_N;
    const int n_in = in_band - cluster_m * TB_SWIZZLE_N;

    const int tile_n = band * TB_SWIZZLE_N + n_in;
    const int tile_m = cluster_m * CLUSTER_SIZE + cta_rank;

    return make_int2(tile_n, tile_m);
}


__global__ 
__cluster_dims__(CLUSTER_SIZE, 1, 1)
__launch_bounds__(THREAD_BLOCK_SIZE)
void kernel(bf16 *A, bf16 *B, bf16 *C, int M, int N, int K,
            const __grid_constant__ CUtensorMap A_tmap, const __grid_constant__ CUtensorMap B_tmap) {
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int total_bids = gridDim.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int cta_rank; // which cluster is this cta in
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    constexpr int16_t cta_mask = (1 << 2) - 1;  // 0b11 in binary, multicasts to both ctas in cluster
    
    int grid_m = M / BLOCK_M;
    int grid_n = N / BLOCK_N;
    int num_tiles = grid_m * grid_n;
    int num_k_tiles = K / BLOCK_K;
    
    // each thread block in cluster only loads half of B
    extern __shared__ __align__(1024) char smem_ptr[];
    const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
    constexpr int A_size = BLOCK_M * BLOCK_K * sizeof(bf16);
    constexpr int B_size = (BLOCK_N / CLUSTER_SIZE) * BLOCK_K * sizeof(bf16);
    
    __shared__ int tmem_addr[1]; // tmem address, returned by tcgen05.alloc
    __shared__ uint64_t mbar[STAGES * 2 + 4];
    const int mbar_empty = static_cast<int>(__cvta_generic_to_shared(mbar));
    const int mbar_filled = mbar_empty + STAGES * 8;
    const int mbar_tmem_empty = mbar_filled + STAGES * 8;
    const int mbar_tmem_full = mbar_tmem_empty + 2 * 8;

    // initialize mbarrier and allocate tmem
    if (warp_id == 0 && is_elect_sync()) {
        for (int i = 0; i < STAGES; i++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_empty + i * 8), "r"(1));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_filled + i * 8), "r"(CLUSTER_SIZE));
        }
        for (int i = 0; i < 2; i++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_tmem_empty + i * 8), "r"(CLUSTER_SIZE * 4));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_tmem_full + i * 8), "r"(1));
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    else if (warp_id == 1) {
        // All 32 threads in warp 1 must call tcgen05.alloc together
        // allocates 2 * BLOCK_N columns for pipelining (128 rows are always allocated)
        const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(addr), "r"(2 * BLOCK_N));
    }

    if constexpr (CLUSTER_SIZE > 1) {
        // visible to all threads in a cluster
        asm volatile("barrier.cluster.arrive.release.aligned;");
        asm volatile("barrier.cluster.wait.acquire.aligned;");
    }
    else {
    __syncthreads();
    }

    const int taddr = tmem_addr[0];  // read the tensor memory address

    if (warp_id == 0 && is_elect_sync()) { // tma warp
        int tma_idx = 0;
        int tma_phase = 1;
        for (int cur_bid = bid; cur_bid < num_tiles; cur_bid += total_bids) {
            int2 tile = compute_position(cur_bid, cta_rank, grid_m, grid_n);
            int tile_m = tile.y;
            int tile_n = tile.x;
            int off_m = tile_m * BLOCK_M;
            int off_n = tile_n * BLOCK_N + cta_rank * CLUSTER_N;

            // wait for compute to eat stage before issuing next tma
            for (int t = 0; t < num_k_tiles; t++) {
                mbarrier_wait(mbar_empty + tma_idx * 8, tma_phase);
                int cta0_mbar_filled = (mbar_filled + tma_idx * 8) & 0xFEFFFFFF; // trick instead of mapa to designate cta0's mbar
                
                // select TMA buffer
                const int A_smem = smem + tma_idx * (A_size + B_size);
                const int B_smem = A_smem + A_size;
                
                // 3D TMA: coordinates are (0, row_offset, k_tile_idx)
                // where k_tile_idx = t * BLOCK_K / 64 = t (when BLOCK_K = 64)
                int k_tile = t * BLOCK_K / 64;
                tma_3d_gmem2smem<CLUSTER_SIZE>(A_smem, &A_tmap, 0, off_m, k_tile, cta0_mbar_filled);
                tma_3d_gmem2smem<CLUSTER_SIZE>(B_smem, &B_tmap, 0, off_n, k_tile, cta0_mbar_filled);
                
                constexpr int copy_size = (BLOCK_M + CLUSTER_N) * BLOCK_K * sizeof(bf16);
                asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                    :: "r"(cta0_mbar_filled), "r"(copy_size) : "memory");
                    
                tma_idx++;
                if (tma_idx == STAGES) {
                    tma_idx = 0;
                    tma_phase ^= 1;
                }
            }
        }
    }
    else if (cta_rank == 0 && warp_id == 1 && is_elect_sync()) { // mma warp (FEED ME)
        int mma_idx = 0;
        int mma_phase = 0;
        int tmem_idx = 0;
        int tmem_phase = 1;
        for (int cur_bid = bid; cur_bid < num_tiles; cur_bid += total_bids) {
            // wait on tmem_empty, signalled from epilogue warps
            mbarrier_wait(mbar_tmem_empty + tmem_idx * 8, tmem_phase);
            for (int t = 0; t < num_k_tiles; t++) {
                // wait for tma to fill stage before issuing mma
                mbarrier_wait(mbar_filled + mma_idx * 8, mma_phase);
                asm volatile("tcgen05.fence::after_thread_sync;");

                const int A_smem = smem + mma_idx * (A_size + B_size);
                const int B_smem = A_smem + A_size;
                const int tmem = taddr + tmem_idx * BLOCK_N;

                // Manually unroll 1st iteration to disable accumulation
                uint64_t a_desc = make_smem_desc_swizzled(A_smem);
                uint64_t b_desc = make_smem_desc_swizzled(B_smem);
                tcgen05_mma_f16<CLUSTER_SIZE>(tmem, a_desc, b_desc, IDESC, t);
                for (int k2 = 1; k2 < CORE_MATRIX_WIDTH / MMA_K; k2++) {
                    int a_addr = A_smem + k2 * MMA_K * 2;
                    int b_addr = B_smem + k2 * MMA_K * 2;
                    uint64_t a_desc = make_smem_desc_swizzled(a_addr);
                    uint64_t b_desc = make_smem_desc_swizzled(b_addr);
                    tcgen05_mma_f16<CLUSTER_SIZE>(tmem, a_desc, b_desc, IDESC, 1);
                }
                // k1 selects the (BLOCK_M, 128B) tile.
                // k2 selects the (BLOCK_M, 32B) tile, whose rows are swizzled.
                for (int k1 = 1; k1 < BLOCK_K / CORE_MATRIX_WIDTH; k1++) {
                    for (int k2 = 0; k2 < CORE_MATRIX_WIDTH / MMA_K; k2++) {
                        int a_addr = A_smem + k1 * CORE_MATRIX_WIDTH_BYTES * BLOCK_M + k2 * MMA_K * 2;
                        int b_addr = B_smem + k1 * CORE_MATRIX_WIDTH_BYTES * CLUSTER_N + k2 * MMA_K * 2;
                        uint64_t a_desc = make_smem_desc_swizzled(a_addr);
                        uint64_t b_desc = make_smem_desc_swizzled(b_addr);
                        tcgen05_mma_f16<CLUSTER_SIZE>(tmem, a_desc, b_desc, IDESC, 1);
                    }
                }
                
                asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                    :: "r"(mbar_empty + mma_idx * 8), "h"(cta_mask) : "memory");
                mma_idx++;
                if (mma_idx == STAGES) {
                    mma_idx = 0;
                    mma_phase ^= 1;
                }
            }
            // signal tmem_full to epilogue warps
            asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                :: "r"(mbar_tmem_full + tmem_idx * 8), "h"(cta_mask) : "memory");
            tmem_idx++;
            if (tmem_idx == 2) {
                tmem_idx = 0;
                tmem_phase ^= 1;
            }
        }
        
    }
    else if (warp_id >= 2) { // 4 epilogue warps
        int epi_warp_id = warp_id % 4;  // warp_id-2 DOES NOT WORK, found with lots of pain
        int epi_idx = 0;
        int epi_phase = 0;
        for (int cur_bid = bid; cur_bid < num_tiles; cur_bid += total_bids) {
            int2 tile = compute_position(cur_bid, cta_rank, grid_m, grid_n);
            int tile_m = tile.y;
            int tile_n = tile.x;
            // output tile offsets in global memory
            int off_m = tile_m * BLOCK_M;
            int off_n = tile_n * BLOCK_N;

            // wait on tmem_full, signalled from mma warp
            mbarrier_wait(mbar_tmem_full + epi_idx * 8, epi_phase);
            asm volatile("tcgen05.fence::after_thread_sync;");

            // tmem -> regs -> gmem
            // Layout D: each epilogue warp handles 32 rows of the 128-row output
            for (int n = 0; n < BLOCK_N / 8; n++) {
                // tmem address tcgen05.ld uses is for the logical output tile produced by the cluster, not the per-cta private 128xBLOCK_N
                int row_base = (cta_rank * 128) + epi_warp_id * 32;   // base row for this warp's 32 lanes
                // tmem address: upper 16 bits = row, lower 16 bits = column
                const int addr = taddr + (row_base << 16) + (epi_idx * BLOCK_N + n * 8);

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

                // Write 16 bytes (8 bf16) to global memory C.
                // Each lane in the epilogue warp owns one row within its 32-row slice.
                int row_in_tile = epi_warp_id * 32 + lane_id;  // 0..127 across 4 warps
                bf16 *out_ptr = C + (off_m + row_in_tile) * N + (off_n + n * 8);
                reinterpret_cast<int4 *>(out_ptr)[0] = reinterpret_cast<int4 *>(out)[0];
            }

            // signal tmem_empty to mma warp, all epilogue warps report to cta 0 mbarrier
            if (is_elect_sync()) {
                int mbar_addr = (mbar_tmem_empty + epi_idx * 8) & 0xFEFFFFFF;
                asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" :: "r"(mbar_addr) : "memory");
            }
            epi_idx++;
            if (epi_idx == 2) {
                epi_idx = 0;
                epi_phase ^= 1;
            }
        }
    }

    if constexpr (CLUSTER_SIZE > 1) {
        // visible to all threads in a cluster
        asm volatile("barrier.cluster.arrive.release.aligned;");
        asm volatile("barrier.cluster.wait.acquire.aligned;");
    }
    else {
        __syncthreads();
    }
    if (warp_id == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
            :: "r"(taddr), "r"(2 * BLOCK_N));
    }
}

static inline void sleep_ms(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

__host__ double run_benchmark(int M, int N, int K, bool ncu = false) {
    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Kernel constants: BLOCK_M=" << BLOCK_M
              << " BLOCK_N=" << BLOCK_N
              << " BLOCK_K=" << BLOCK_K
              << " CLUSTER_SIZE=" << CLUSTER_SIZE
              << " STAGES=" << STAGES << "\n";

    if ((M % BLOCK_M) != 0 || (N % BLOCK_N) != 0 || (K % BLOCK_K) != 0) {
        std::cout << "Skipping: dimensions must be divisible by BLOCK_M/BLOCK_N/BLOCK_K\n";
        return 0.0;
    }

    // Cooldown between configurations
    sleep_ms(500);

    // L2 cache eviction - multiple buffer groups (match ThunderKittens pattern)
    int l2_cache_size = 0;
    CUDACHECK(cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0));
    const size_t arg_bytes =
        2ull * (size_t(M) * size_t(K) + size_t(N) * size_t(K) + size_t(M) * size_t(N)) * sizeof(bf16);
    const size_t ideal_arg_bytes = size_t(l2_cache_size) * 3ull;
    const int arg_group_count = (arg_bytes > ideal_arg_bytes) ? 1 : int(ideal_arg_bytes / arg_bytes) + 1;

    // Allocate device memory
    std::vector<bf16*> d_A(arg_group_count);
    std::vector<bf16*> d_B(arg_group_count);
    std::vector<bf16*> d_C(arg_group_count);
    bf16* d_C_ref = nullptr;
    for (int i = 0; i < arg_group_count; i++) {
        CUDACHECK(cudaMalloc(&d_A[i], size_t(M) * size_t(K) * sizeof(bf16)));
        CUDACHECK(cudaMalloc(&d_B[i], size_t(N) * size_t(K) * sizeof(bf16))); // stored as N×K (B column-major view)
        CUDACHECK(cudaMalloc(&d_C[i], size_t(M) * size_t(N) * sizeof(bf16)));
    }
    CUDACHECK(cudaMalloc(&d_C_ref, size_t(M) * size_t(N) * sizeof(bf16)));
    std::cout << "Allocated device memory (groups=" << arg_group_count << ")\n";

    // Initialize matrices with random values on device
    uint64_t seed = 2024;
    for (int i = 0; i < arg_group_count; i++) {
        fill_random(d_A[i], size_t(M) * size_t(K), seed + i * 100);
        fill_random(d_B[i], size_t(N) * size_t(K), seed + i * 100 + 1);
    }
    CUDACHECK(cudaDeviceSynchronize());
    std::cout << "Initialized A/B on device\n";

    // Compute reference GEMM on device (B is column-major storage: N×K)
    reference_gemm_Bcolmaj(d_C_ref, d_A[0], d_B[0], M, N, K);
    CUDACHECK(cudaDeviceSynchronize());
    std::cout << "Computed reference GEMM on device\n";

    // Grid: persistent kernel uses NUM_SM blocks
    constexpr int grid = NUM_SM;
    dim3 block(THREAD_BLOCK_SIZE);

    // Create 3D tensor maps per buffer group (because pointers differ)
    // 3D view: (64, height, K/64) with appropriate strides for 128B swizzle
    std::vector<CUtensorMap> A_tmap(arg_group_count);
    std::vector<CUtensorMap> B_tmap(arg_group_count);
    for (int i = 0; i < arg_group_count; i++) {
        CUresult res1 = init_tmap_3d_AB(&A_tmap[i], d_A[i], M, K, BLOCK_M, BLOCK_K);
        CUresult res2 = init_tmap_3d_AB(&B_tmap[i], d_B[i], N, K, CLUSTER_N, BLOCK_K);
        if (res1 != CUDA_SUCCESS || res2 != CUDA_SUCCESS) {
            std::cerr << "init_tmap_3d_AB failed\n";
            std::exit(EXIT_FAILURE);
        }
    }

    // Calculate shared memory size for multi-stage pipeline
    int size_AB = (BLOCK_M + BLOCK_N / CLUSTER_SIZE) * BLOCK_K * STAGES;
    int smem_size = size_AB * sizeof(bf16);
    if (smem_size > 48'000) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    // Number of iterations
    int num_warmups = ncu ? 0 : 500;
    int num_iters = ncu ? 1 : 100;

    // Warmup
    for (int i = 0; i < num_warmups; i++) {
        int idx = i % arg_group_count;
        kernel<<<grid, block, smem_size>>>(d_A[idx], d_B[idx], d_C[idx], M, N, K, A_tmap[idx], B_tmap[idx]);
    }
    CUDACHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        int idx = i % arg_group_count;
        kernel<<<grid, block, smem_size>>>(d_A[idx], d_B[idx], d_C[idx], M, N, K, A_tmap[idx], B_tmap[idx]);
    }
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    // Calculate duration and TFLOPs
    float milliseconds = 0.0f;
    CUDACHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    double microseconds = milliseconds * 1000.0 / num_iters;
    double flops = 2.0 * double(M) * double(N) * double(K);
    double tflops = (flops / microseconds) / 1e6;
    std::cout << "Average kernel execution time: " << microseconds << " us\n";
    std::cout << "Performance: " << tflops << " TFLOPs\n";

    // Verify results (compare group 0 output to reference)
    check_correctness(d_C[0], d_C_ref, size_t(M) * size_t(N));

    // Clean up
    for (int i = 0; i < arg_group_count; i++) {
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
    }
    cudaFree(d_C_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return tflops;
}

int main() {
    bool ncu = false;

    int N;
    N = 1024;
    run_benchmark(N, N, N, ncu);
    N = 2048;
    run_benchmark(N, N, N, ncu);
    N = 4096;
    run_benchmark(N, N, N, ncu);
    N = 8192;
    run_benchmark(N, N, N, ncu);
    N = 16384;
    run_benchmark(N, N, N, ncu);

    return 0;
}
