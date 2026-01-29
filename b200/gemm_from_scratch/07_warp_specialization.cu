/* 
warp specialization
929 tflops

warp specialization splits roles into different warps inside the same CTA, 
so each warp can run a tight loop doing only its job, instead of all warps 
running one mixed loop with “if I’m the load warp do load else do compute” checks.
stop relying on the compiler to perfectly interleave unrelated instruction streams;
give the scheduler two independent warps to run in parallel and coordinate 
with cheap shared memory barriers

also changed epilogue store from registers->gmem to registers->smem->gmem,
for coalesced global memory writes. before: each thread wrote its own row,
so 32 threads in a warp hit 32 different rows = uncoalesced. now: part 1
stages bf16 results into shared mem with padding (264 cols/row avoids bank
conflicts), part 2 has each warp write one row of smem at a time with 32 lanes
writing 32 consecutive 16B segments = perfectly coalesced. this added >40 tflops!

Layout convention:
  - A is M×K, row-major:     A[m,k] = A_storage[m * K + k]
  - B is K×N, column-major:  B[k,n] = B_storage[n * K + k]  (stored as N×K)
  - C is M×N, row-major:     C[m,n] = C_storage[m * N + n]

OUTLINE:
for num_stages = 2
mbars_filled[2]
mbars_empty[2]

tma warp: waiting on mbar empty
for k = 0; k < numstages 
    issue tma(mbar filled)
then for k = num stages; k < k_tiles:
    wait on mbar empty, issue tma(mbar filled)

mma warp: waiting on mbar filled
mma_phase = 1
for t in k_tiles
    wait on mbar filled
    call mma, and commit on mbar empty to signal it when mma finishes
*/

#include "boilerplate.cuh"
#include "helpers.cuh"

using bf16 = __nv_bfloat16;

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;
constexpr int THREAD_BLOCK_SIZE = WARP_SIZE * NUM_WARPS;
constexpr int STAGES = 4;

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 256;
constexpr int BLOCK_K = 64;
constexpr uint32_t CORE_MATRIX_WIDTH_BYTES = 128; // swizzle bytes change width
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
    
    int tile_m = blockIdx.x; 
    int tile_n = blockIdx.y;
    int num_k_tiles = K / BLOCK_K;

    // unified shared memory pool: reuse smem between TMA/MMA phase (A+B tiles) and epilogue (C staging)
    // size is max of (A+B) or C since they're used at different times
    constexpr int A_ELEMS = BLOCK_M * BLOCK_K * STAGES;
    constexpr int B_ELEMS = BLOCK_N * BLOCK_K * STAGES;
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
        B_smem[i] = static_cast<int>(__cvta_generic_to_shared(B_tile + i * (BLOCK_N * BLOCK_K)));
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
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_filled[i]), "r"(1));
        }
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_main[0]), "r"(1));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    else if (warp_id == 1) {
        // All 32 threads in warp 1 must call tcgen05.alloc together
        // It allocates BLOCK_N columns of tensor memory (128 rows are always allocated)
        const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" 
            :: "r"(addr), "r"(BLOCK_N));
    }
    __syncthreads();  // make mbarrier init and tmem allocation visible to all threads

    // read the tensor memory address
    const int taddr = tmem_addr[0];

    if (warp_id == 0 && is_elect_sync()) { // tma warp
        int off_m = tile_m * BLOCK_M;
        int off_n = tile_n * BLOCK_N;

        // wait for compute to eat stage before issuing next tma
        int tma_phase = 1; // initialized to 1 since all stages initially empty 
        int tma_idx = 0;
        for (int t = 0; t < num_k_tiles; t++) {
            mbarrier_wait(mbar_empty[tma_idx], tma_phase);
            for (int k = 0; k < BLOCK_K / CORE_MATRIX_WIDTH; k++) {
                int off_k = t * BLOCK_K + k * CORE_MATRIX_WIDTH;
                tma_2d_gmem2smem(A_smem[tma_idx] + k * BLOCK_M * CORE_MATRIX_WIDTH_BYTES, &A_tmap, off_k, off_m, mbar_filled[tma_idx]);
                tma_2d_gmem2smem(B_smem[tma_idx] + k * BLOCK_N * CORE_MATRIX_WIDTH_BYTES, &B_tmap, off_k, off_n, mbar_filled[tma_idx]);
            }
            constexpr int copy_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(bf16);
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(mbar_filled[tma_idx]), "r"(copy_size) : "memory");
                
            tma_idx++;
            if (tma_idx == STAGES) {
                tma_idx = 0;
                tma_phase ^= 1;
            }
        }
    }
    else if (warp_id == 1 && is_elect_sync()) { // mma warp (HUNGRY)
        int mma_phase = 0;
        for (int t = 0; t < num_k_tiles; t++) {
            int mma_idx = t % STAGES;
            // wait for tma to fill stage before issuing mma
            mbarrier_wait(mbar_filled[mma_idx], mma_phase);
            asm volatile("tcgen05.fence::after_thread_sync;");

            // Manually unroll 1st iteration to disable accumulation
            uint64_t a_desc = make_smem_desc_swizzled(A_smem[mma_idx]);
            uint64_t b_desc = make_smem_desc_swizzled(B_smem[mma_idx]);
            tcgen05_mma_f16(taddr, a_desc, b_desc, IDESC, t);
            for (int k2 = 1; k2 < CORE_MATRIX_WIDTH / MMA_K; k2++) {
                int a_addr = A_smem[mma_idx] + k2 * MMA_K * 2;
                int b_addr = B_smem[mma_idx] + k2 * MMA_K * 2;
                uint64_t a_desc = make_smem_desc_swizzled(a_addr);
                uint64_t b_desc = make_smem_desc_swizzled(b_addr);
                tcgen05_mma_f16(taddr, a_desc, b_desc, IDESC, 1);
            }
            // k1 selects the (BLOCK_M, 128B) tile.
            // k2 selects the (BLOCK_M, 32B) tile, whose rows are swizzled.
            for (int k1 = 1; k1 < BLOCK_K / CORE_MATRIX_WIDTH; k1++) {
                for (int k2 = 0; k2 < CORE_MATRIX_WIDTH / MMA_K; k2++) {
                    int a_addr = A_smem[mma_idx] + k1 * CORE_MATRIX_WIDTH_BYTES * BLOCK_M + k2 * MMA_K * 2;
                    int b_addr = B_smem[mma_idx] + k1 * CORE_MATRIX_WIDTH_BYTES * BLOCK_N + k2 * MMA_K * 2;
                    uint64_t a_desc = make_smem_desc_swizzled(a_addr);
                    uint64_t b_desc = make_smem_desc_swizzled(b_addr);
                    tcgen05_mma_f16(taddr, a_desc, b_desc, IDESC, 1);
                }
            }
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                :: "r"(mbar_empty[mma_idx]) : "memory");
            if (mma_idx == STAGES - 1) {
                mma_phase ^= 1;
            }
        }
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
            :: "r"(mbar_main[0]) : "memory");
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
        int row_base = warp_id * 32;   // base row for this warp's 32 lanes
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
    dim3 grid(M / BLOCK_M, N / BLOCK_N);
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
