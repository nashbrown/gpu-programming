/* 
cluster mma
638 tflops

2 thread blocks in a cluster
issue an mma together using distributed shared memory of their peer thread block

Layout convention:
  - A is M×K, row-major:     A[m,k] = A_storage[m * K + k]
  - B is K×N, column-major:  B[k,n] = B_storage[n * K + k]  (stored as N×K)
  - C is M×N, row-major:     C[m,n] = C_storage[m * N + n]

*/

#include "boilerplate.cuh"
#include "helpers.cuh"

using bf16 = __nv_bfloat16;

#include <thread>
#include <chrono>

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;
constexpr int THREAD_BLOCK_SIZE = WARP_SIZE * NUM_WARPS;
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
    | ((MMA_M / 16) << 24U);             // bits 24-30: MMA_M = 128/16 = 8


__global__ 
__cluster_dims__(CLUSTER_SIZE, 1, 1)
__launch_bounds__(THREAD_BLOCK_SIZE)
void kernel(bf16 *A, bf16 *B, bf16 *C, int M, int N, int K,
            const __grid_constant__ CUtensorMap A_tmap, const __grid_constant__ CUtensorMap B_tmap) {
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int cta_rank; // which cluster is this cta in
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    constexpr int16_t cta_mask = (1 << 2) - 1;  // 0b11 in binary, multicasts to both ctas in cluster
    
    const int grid_m = M / BLOCK_M;
    const int grid_n = N / BLOCK_N;
    // pair cta's must have adjacent tile_m and same tile_n
    const int tile_m = bid / (grid_n * CLUSTER_SIZE) * CLUSTER_SIZE + (bid % CLUSTER_SIZE);
    const int tile_n = (bid / CLUSTER_SIZE) % grid_n;
    int num_k_tiles = K / BLOCK_K;

    // unified shared memory pool: reuse smem between TMA/MMA phase (A+B tiles) and epilogue (C staging)
    // size is max of (A+B) or C since they're used at different times
    constexpr int A_ELEMS = BLOCK_M * BLOCK_K * STAGES;
    constexpr int B_ELEMS = BLOCK_N / CLUSTER_SIZE * BLOCK_K * STAGES;
    constexpr int EPILOGUE_PAD = 8;
    constexpr int EPILOGUE_LD  = BLOCK_N + EPILOGUE_PAD;
    constexpr int C_ELEMS = BLOCK_M * EPILOGUE_LD;

    constexpr int SMEM_ELEMS = (A_ELEMS + B_ELEMS > C_ELEMS) ? (A_ELEMS + B_ELEMS) : C_ELEMS;

    __shared__ alignas(1024) bf16 smem_pool[SMEM_ELEMS];

    bf16* A_tile = smem_pool;
    bf16* B_tile = smem_pool + A_ELEMS;

    int A_smem[STAGES];
    int B_smem[STAGES];
    for (int i = 0; i < STAGES; i++) {
        A_smem[i] = static_cast<int>(__cvta_generic_to_shared(A_tile + i * (BLOCK_M * BLOCK_K)));
        B_smem[i] = static_cast<int>(__cvta_generic_to_shared(B_tile + i * (CLUSTER_N * BLOCK_K)));
    }

    __shared__ uint64_t mbar[STAGES * 2 + 1];
    int mbar_empty[STAGES];
    int mbar_filled[STAGES];
    int mbar_main[1];
    for (int i = 0; i < STAGES; i++) {
        mbar_empty[i] = static_cast<int>(__cvta_generic_to_shared(mbar + i * 2));
        mbar_filled[i] = static_cast<int>(__cvta_generic_to_shared(mbar + i * 2 + 1));
    }
    mbar_main[0] = static_cast<int>(__cvta_generic_to_shared(mbar + STAGES * 2));
    __shared__ int tmem_addr[1]; // tmem address, returned by tcgen05.alloc

    // initialize mbarrier and allocate tmem
    if (warp_id == 0 && is_elect_sync()) {
        for (int i = 0; i < STAGES; i++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_empty[i]), "r"(1));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_filled[i]), "r"(CLUSTER_SIZE));
        }
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_main[0]), "r"(1));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    else if (warp_id == 1) {
        // All 32 threads in warp 1 must call tcgen05.alloc together
        // It allocates BLOCK_N columns of tensor memory (128 rows are always allocated)
        const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(addr), "r"(BLOCK_N));
    }
    __syncthreads();  // make mbarrier init and tmem allocation visible to all threads
    // cluster wide sync necessary: check-in, then stop and wait for peer ctas
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;"   ::: "memory");

    const int taddr = tmem_addr[0]; // read the tensor memory address

    if (warp_id == 0 && is_elect_sync()) { // tma warp
        int off_m = tile_m * BLOCK_M;
        int off_n = tile_n * BLOCK_N + cta_rank * CLUSTER_N;

        // wait for compute to eat stage before issuing next tma
        int tma_phase = 1;
        int tma_idx = 0;
        for (int t = 0; t < num_k_tiles; t++) {
            mbarrier_wait(mbar_empty[tma_idx], tma_phase);
            int cta0_mbar_filled = mbar_filled[tma_idx] & 0xFEFFFFFF; // trick instead of mapa to designate cta0's mbar
            for (int k = 0; k < BLOCK_K / CORE_MATRIX_WIDTH; k++) {
                int off_k = t * BLOCK_K + k * CORE_MATRIX_WIDTH;
                tma_2d_gmem2smem<CLUSTER_SIZE>(A_smem[tma_idx] + k * BLOCK_M * CORE_MATRIX_WIDTH_BYTES, &A_tmap, off_k, off_m, cta0_mbar_filled);
                tma_2d_gmem2smem<CLUSTER_SIZE>(B_smem[tma_idx] + k * CLUSTER_N * CORE_MATRIX_WIDTH_BYTES, &B_tmap, off_k, off_n, cta0_mbar_filled);
            }
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
    else if (cta_rank == 0 && warp_id == 1 && is_elect_sync()) { // mma warp (HUNGRY)
        int mma_phase = 0;
        for (int t = 0; t < num_k_tiles; t++) {
            int mma_idx = t % STAGES;
            // wait for tma to fill stage before issuing mma
            mbarrier_wait(mbar_filled[mma_idx], mma_phase);
            asm volatile("tcgen05.fence::after_thread_sync;");

            // Manually unroll 1st iteration to disable accumulation
            uint64_t a_desc = make_smem_desc_swizzled(A_smem[mma_idx]);
            uint64_t b_desc = make_smem_desc_swizzled(B_smem[mma_idx]);
            tcgen05_mma_f16<CLUSTER_SIZE>(taddr, a_desc, b_desc, IDESC, t);
            for (int k2 = 1; k2 < CORE_MATRIX_WIDTH / MMA_K; k2++) {
                int a_addr = A_smem[mma_idx] + k2 * MMA_K * 2;
                int b_addr = B_smem[mma_idx] + k2 * MMA_K * 2;
                uint64_t a_desc = make_smem_desc_swizzled(a_addr);
                uint64_t b_desc = make_smem_desc_swizzled(b_addr);
                tcgen05_mma_f16<CLUSTER_SIZE>(taddr, a_desc, b_desc, IDESC, 1);
            }
            // k1 selects the (BLOCK_M, 128B) tile.
            // k2 selects the (BLOCK_M, 32B) tile, whose rows are swizzled.
            for (int k1 = 1; k1 < BLOCK_K / CORE_MATRIX_WIDTH; k1++) {
                for (int k2 = 0; k2 < CORE_MATRIX_WIDTH / MMA_K; k2++) {
                    int a_addr = A_smem[mma_idx] + k1 * CORE_MATRIX_WIDTH_BYTES * BLOCK_M + k2 * MMA_K * 2;
                    int b_addr = B_smem[mma_idx] + k1 * CORE_MATRIX_WIDTH_BYTES * CLUSTER_N + k2 * MMA_K * 2;
                    uint64_t a_desc = make_smem_desc_swizzled(a_addr);
                    uint64_t b_desc = make_smem_desc_swizzled(b_addr);
                    tcgen05_mma_f16<CLUSTER_SIZE>(taddr, a_desc, b_desc, IDESC, 1);
                }
            }
            
            asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                :: "r"(mbar_empty[mma_idx]), "h"(cta_mask) : "memory");
            if (mma_idx == STAGES - 1) {
                mma_phase ^= 1;
            }
        }
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
            :: "r"(mbar_main[0]), "h"(cta_mask) : "memory");
    }
    __syncthreads();
    // can pass syncthreads after last mma has been issued, but not finished, so use one last mbarrier
    mbarrier_wait(mbar_main[0], 0);

    // don't let my later tcgen05.ld get reordered before the wait we just did
    asm volatile("tcgen05.fence::after_thread_sync;");

    // Output tile offsets in global memory
    int off_m = tile_m * BLOCK_M;
    int off_n = tile_n * BLOCK_N;

    bf16* C_smem = smem_pool;

    // Each thread owns one row: r = tid (0..127)
    int r = tid;

    // part 1: tmem -> regs -> shared (row-major with padded LD)
    // We load 8 FP32 at a time for each row segment (col = n*8), convert to 8 bf16, store 16B into smem.
    for (int n = 0; n < BLOCK_N / 8; n++) {
        // tmem address: upper 16 bits = row, lower 16 bits = column
        int row_base = (cta_rank * 128) + warp_id * 32;   // base row for this warp's 32 lanes
        int col = n * 8;
        int addr = taddr + (row_base << 16) + col;

        float tmp[8];
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
              "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
            : "r"(addr)
        );
        asm volatile("tcgen05.wait::ld.sync.aligned;");

        // Pack into 8 bf16 (as 4x bfloat162 = 16B)
        nv_bfloat162 out[4];
        for (int i = 0; i < 4; i++) {
            out[i] = __float22bfloat162_rn({tmp[i * 2], tmp[i * 2 + 1]});
        }

        // Write 16B to shared: C_smem[r][col:col+7]
        int smem_idx = r * EPILOGUE_LD + col; // in bf16 elements
        *reinterpret_cast<int4*>(&C_smem[smem_idx]) =
            *reinterpret_cast<const int4*>(out);
    }

    __syncthreads();

    // part 2: shared -> global, fully coalesced
    int lane = lane_id;
    constexpr int SEG_ELEMS = 8;                 // 8 bf16 = 16B per int4
    const int num_segs = BLOCK_N / SEG_ELEMS;    // number of 16B segments per row

    for (int row_base = 0; row_base < BLOCK_M; row_base += NUM_WARPS) {
        int rr = row_base + warp_id;

        // handle any BLOCK_N <= 256 (and also works for 256)
        for (int seg = lane; seg < num_segs; seg += WARP_SIZE) {
            int col = seg * SEG_ELEMS;

            int smem_idx = rr * EPILOGUE_LD + col;
            bf16* gptr   = C + (off_m + rr) * N + (off_n + col);

            *reinterpret_cast<int4*>(gptr) =
                *reinterpret_cast<const int4*>(&C_smem[smem_idx]);
        }
    }

    // ensure threads finish reading from tmem before deallocation
    __syncthreads();
    // cluster wide sync, check-in, then stop and wait for peer ctas
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;"   ::: "memory");
    if (warp_id == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
            :: "r"(taddr), "r"(BLOCK_N));
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

    // Grid: one block per output tile
    int grid = (M / BLOCK_M) * (N / BLOCK_N);
    dim3 block(THREAD_BLOCK_SIZE);

    // Create tensor maps per buffer group (because pointers differ)
    std::vector<CUtensorMap> A_tmap(arg_group_count);
    std::vector<CUtensorMap> B_tmap(arg_group_count);
    for (int i = 0; i < arg_group_count; i++) {
        CUresult res1 = init_tmap_2d(&A_tmap[i], d_A[i], K, M, CORE_MATRIX_WIDTH, BLOCK_M, CU_TENSOR_MAP_SWIZZLE_128B);
        CUresult res2 = init_tmap_2d(&B_tmap[i], d_B[i], K, N, CORE_MATRIX_WIDTH, CLUSTER_N, CU_TENSOR_MAP_SWIZZLE_128B);
        if (res1 != CUDA_SUCCESS || res2 != CUDA_SUCCESS) {
            std::cerr << "init_tmap_2d failed\n";
            std::exit(EXIT_FAILURE);
        }
    }

    // Number of iterations
    int num_warmups = ncu ? 0 : 500;
    int num_iters   = ncu ? 1 : 100;

    // Warmup
    for (int i = 0; i < num_warmups; i++) {
        int idx = i % arg_group_count;
        kernel<<<grid, block>>>(d_A[idx], d_B[idx], d_C[idx], M, N, K, A_tmap[idx], B_tmap[idx]);
    }
    CUDACHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        int idx = i % arg_group_count;
        kernel<<<grid, block>>>(d_A[idx], d_B[idx], d_C[idx], M, N, K, A_tmap[idx], B_tmap[idx]);
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
