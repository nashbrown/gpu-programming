#include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap
#include <cuda_bf16.h>
#include <cassert>

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
    // Get pointer to cuTensorMapEncodeTiled
    cudaDriverEntryPointQueryResult driver_status;
    void* cuTensorMapEncodeTiled_ptr = nullptr;
    CUDACHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, cudaEnableDefault, &driver_status));
    assert(driver_status == cudaDriverEntryPointSuccess);
  
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}

// gmem_width, smem_width in elements, not bytes
CUresult init_tmap_2d(CUtensorMap* tmap, nv_bfloat16* tensor_ptr, uint64_t gmem_width, uint64_t gmem_height, uint32_t smem_width, uint32_t smem_height, CUtensorMapSwizzle swizzle) {
    auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
    // rank is the number of dimensions of the array.
    constexpr uint32_t rank = 2;
    uint64_t size[rank] = {gmem_width, gmem_height};
    // The stride is the number of bytes to traverse from the first element of one row to the next.
    // It must be a multiple of 16.
    uint64_t stride[rank - 1] = {gmem_width * sizeof(nv_bfloat16)};
    // The box_size is the size of the shared memory buffer that is used as the
    // destination of a TMA transfer.
    uint32_t box_size[rank] = {smem_width, smem_height};
    // The distance between elements in units of sizeof(element). A stride of 2
    // can be used to load only the real component of a complex-valued tensor, for instance.
    uint32_t elem_stride[rank] = {1, 1};
    CUresult res = cuTensorMapEncodeTiled(
        tmap,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        rank,                       // cuuint32_t tensorRank,
        tensor_ptr,                 // void *globalAddress,
        size,                       // const cuuint64_t *globalDim,
        stride,                     // const cuuint64_t *globalStrides,
        box_size,                   // const cuuint32_t *boxDim,
        elem_stride,                // const cuuint32_t *elementStrides,
        // Interleave patterns can be used to accelerate loading of values that
        // are less than 4 bytes long.
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        // Swizzling can be used to avoid shared memory bank conflicts.
        swizzle,
        // L2 Promotion can be used to widen the effect of a cache-policy to a wider
        // set of L2 cache lines.
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        // Any element that is outside of bounds will be set to zero by the TMA transfer.
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    return res;
}

// 3D tensor map for tcgen05 MMA input layout
// Input layout: contiguous blocks of (smem_height, 64) elements, with 128B swizzling
// 3D view: (64, gmem_height, gmem_K / 64) with strides (gmem_K * 2, 128) bytes
// TMA coordinates: (0, row_offset, k_tile_idx) where k_tile_idx = k_offset / 64
CUresult init_tmap_3d_AB(CUtensorMap* tmap, nv_bfloat16* tensor_ptr, 
                          uint64_t gmem_height, uint64_t gmem_K,
                          uint32_t smem_height, uint32_t smem_K) {
    auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
    constexpr uint32_t rank = 3;
    // globalDim: (64, height, K/64) - 64-element wide chunks, height rows, K/64 chunks
    uint64_t globalDim[rank] = {64, gmem_height, gmem_K / 64};
    // globalStrides: stride[0] = row stride in bytes, stride[1] = K-chunk stride in bytes
    uint64_t globalStrides[rank - 1] = {gmem_K * sizeof(nv_bfloat16), 128};
    // boxDim: (64, smem_height, smem_K/64) - load 64 elements × smem_height rows × smem_K/64 chunks
    uint32_t boxDim[rank] = {64, smem_height, smem_K / 64};
    uint32_t elementStrides[rank] = {1, 1, 1};
    
    CUresult res = cuTensorMapEncodeTiled(
        tmap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        rank,
        tensor_ptr,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    return res;
}

// this does not elect so the name is misleading
__device__ inline bool is_elected() {
    return threadIdx.x == 0 && threadIdx.y == 0;
}

__device__ inline uint32_t is_elect_sync() {
    uint32_t pred = 0;
    asm volatile(
      "{\n\t"
      ".reg .pred %%px;\n\t"
      "elect.sync _|%%px, %1;\n\t"
      "@%%px mov.s32 %0, 1;\n\t"
      "}"
      : "+r"(pred)
      : "r"(0xFFFFFFFF)
    );
    return pred;
}

template <int CLUSTER_SIZE = 1>
__device__ inline
void tma_2d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::%5 [%0], [%1, {%2, %3}], [%4];"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(mbar_addr), "n"(CLUSTER_SIZE) : "memory");
}

template <int CLUSTER_SIZE = 1>
__device__ inline
void tma_3d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::%6 [%0], [%1, {%2, %3, %4}], [%5];"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "n"(CLUSTER_SIZE) : "memory");
}

__device__ inline
void mbarrier_wait(int mbar_addr, int phase) {
  asm volatile(
    "{\n\t"
    ".reg .pred P1;\n\t"
    "LAB_WAIT:\n\t"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, 0;\n\t"
    "@P1 bra.uni DONE;\n\t"
    "bra.uni LAB_WAIT;\n\t"
    "DONE:\n\t"
    "}"
    :: "r"(mbar_addr), "r"(phase)
  );
}

// Cluster-scope mbarrier wait
__device__ inline
void mbarrier_wait_cluster(int mbar_addr, int phase) {
  uint32_t ticks = 0x989680;
  asm volatile(
    "{\n\t"
    ".reg .pred P1;\n\t"
    "LAB_WAIT:\n\t"
    "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1, %2;\n\t"
    "@P1 bra.uni DONE;\n\t"
    "bra.uni LAB_WAIT;\n\t"
    "DONE:\n\t"
    "}"
    :: "r"(mbar_addr), "r"(phase), "r"(ticks)
  );
}

// Encode a value for shared memory descriptor (mask to 18 bits, divide by 16)
__device__ inline
constexpr uint64_t desc_encode(uint64_t x) { return (x & 0x3'FFFFULL) >> 4ULL; }

// addr: shared memory byte offset
// height: number of rows (BLOCK_M for A, BLOCK_N for B)
__device__ inline
uint64_t make_smem_desc(int addr, int height) {
    int LBO = height * 16;   // Leading Byte Offset: stride between rows (height * 16 bytes)
    int SBO = 8 * 16;        // Stride Byte Offset: stride to next core matrix (8 rows × 16 bytes)
    return desc_encode(addr)                 // bits 0-15:  base address
         | (desc_encode(LBO) << 16)          // bits 16-31: LBO
         | (desc_encode(SBO) << 32)          // bits 32-45: SBO
         | (1ULL << 46);                     // bit 46: flag
}

__device__ inline
uint64_t make_smem_desc_swizzled(int addr) {
    int SBO = 8 * 128;  // 128B swizzle
    return desc_encode(addr)                 // bits 0-15: base address
         | (desc_encode(SBO) << 32ULL)       // bits 32-45: SBO
         | (1ULL << 46ULL)                   // bit 46: flag
         | (2ULL << 61ULL);                  // bits 61-63: swizzle = 128B
}

// taddr: tensor memory address
// a_desc, b_desc: shared memory descriptors for A and B
// i_desc: instruction descriptor (encodes dtypes and MMA shape)
// enable_input_d: 0: D = A @ B, 1: D = A @ B + D (accumulate)
template <int CLUSTER_SIZE = 1>
__device__ inline
void tcgen05_mma_f16(int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::%5.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(enable_input_d), "n"(CLUSTER_SIZE)
    );
}

