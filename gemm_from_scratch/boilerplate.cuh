#ifndef GEMM_COMMON_CUH
#define GEMM_COMMON_CUH

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// For checking CUDA runtime API calls
#define CUDACHECK(cmd) do {                                   \
    cudaError_t err = cmd;                                    \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",    \
            __FILE__, __LINE__, cudaGetErrorString(err));     \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

///////////////////////////////////////////////////////////////////////////////////////////////////
// Type conversion helpers
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename To, typename From>
struct convertor {
    __host__ __device__ static inline To convert(From val) { return (To)val; }
};

// bf16 -> float
template <>
struct convertor<float, __nv_bfloat16> {
    __host__ __device__ static inline float convert(__nv_bfloat16 val) { return __bfloat162float(val); }
};

// float -> bf16
template <>
struct convertor<__nv_bfloat16, float> {
    __host__ __device__ static inline __nv_bfloat16 convert(float val) { return __float2bfloat16(val); }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// GPU random fill kernel
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__global__ void fill_random_kernel(T* data, size_t count, uint64_t seed, float min_val, float max_val) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        // Splitmix64 hash for uniform random bits
        uint64_t x = seed + idx;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x = x ^ (x >> 31);
        // Upper 24 bits to float in [0,1)
        float u = (float)(x >> 40) * (1.0f / 16777216.0f);
        // Scale to [min_val, max_val]
        float val = u * (max_val - min_val) + min_val;
        data[idx] = convertor<T, float>::convert(val);
    }
}

template <typename T>
inline void fill_random(T* data, size_t count, uint64_t seed, float min_val = -1.0f, float max_val = 1.0f) {
    dim3 block(256);
    dim3 grid((count + 255) / 256);
    fill_random_kernel<T><<<grid, block>>>(data, count, seed, min_val, max_val);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// GPU reference GEMM: C = A @ B (row-major, MxK @ KxN = MxN)
// Accumulates in float for accuracy regardless of input type
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__global__ void reference_gemm_kernel(T* C, T const* A, T const* B, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a = convertor<float, T>::convert(A[row * K + k]);
            float b = convertor<float, T>::convert(B[k * N + col]);
            acc += a * b;
        }
        C[row * N + col] = convertor<T, float>::convert(acc);
    }
}

template <typename T>
inline void reference_gemm(T* C, T const* A, T const* B, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    reference_gemm_kernel<T><<<grid, block>>>(C, A, B, M, N, K);
}

// Convenience overload for square matrices
template <typename T>
inline void reference_gemm(T* C, T const* A, T const* B, int N) {
    reference_gemm(C, A, B, N, N, N);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// GPU reference GEMM with B column-major: C = A @ B^T_storage
// A is MxK (row-major), B is stored as NxK (column-major, i.e. transposed storage), C is MxN
// Logically computes: C[m,n] = sum_k A[m,k] * B[k,n] where B[k,n] is accessed as B_colmaj[n,k]
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__global__ void reference_gemm_Bcolmaj_kernel(T* C, T const* A, T const* B, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a = convertor<float, T>::convert(A[row * K + k]);
            // B is stored column-major (NxK layout), so B[k,col] = B_storage[col * K + k]
            float b = convertor<float, T>::convert(B[col * K + k]);
            acc += a * b;
        }
        C[row * N + col] = convertor<T, float>::convert(acc);
    }
}

template <typename T>
inline void reference_gemm_Bcolmaj(T* C, T const* A, T const* B, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    reference_gemm_Bcolmaj_kernel<T><<<grid, block>>>(C, A, B, M, N, K);
}

// Convenience overload for square matrices
template <typename T>
inline void reference_gemm_Bcolmaj(T* C, T const* A, T const* B, int N) {
    reference_gemm_Bcolmaj(C, A, B, N, N, N);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Correctness check (takes device pointers, copies to host for comparison)
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline void check_correctness(T const* d_result, T const* d_ref, size_t count) {
    std::vector<T> h_result(count);
    std::vector<T> h_ref(count);

    cudaMemcpy(h_result.data(), d_result, count * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref.data(), d_ref, count * sizeof(T), cudaMemcpyDeviceToHost);

    double abs_sum = 0.0, abs_max = 0.0;
    double err_sum = 0.0, err_max = 0.0;

    for (size_t i = 0; i < count; ++i) {
        float val = convertor<float, T>::convert(h_result[i]);
        float ref = convertor<float, T>::convert(h_ref[i]);
        float err = std::abs(val - ref);

        abs_sum += std::abs(val);
        abs_max = std::max(abs_max, (double)std::abs(val));
        err_sum += err;
        err_max = std::max(err_max, (double)err);
    }

    double abs_mean = abs_sum / count;
    double err_mean = err_sum / count;

    std::cout << "abs mean: " << std::setw(12) << abs_mean << std::endl;
    std::cout << "abs max:  " << std::setw(12) << abs_max << std::endl;
    std::cout << "err mean: " << std::setw(12) << err_mean << std::endl;
    std::cout << "err max:  " << std::setw(12) << err_max << std::endl;
}

#endif // GEMM_COMMON_CUH
