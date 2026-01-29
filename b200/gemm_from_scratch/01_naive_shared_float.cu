/* 
Plain cuda tiled gemm, uses shared mem
3.5 tflops
*/

#include "boilerplate.cuh"

#define NUM_THREADS 32

__global__ void kernel(float *A, float *B, float *C, int N) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;
    int tile_dim = blockDim.x;
    int num_tiles = N / tile_dim;
    __shared__ float A_tile[NUM_THREADS * NUM_THREADS]; // has to be known at compile time
    __shared__ float B_tile[NUM_THREADS * NUM_THREADS];
    float acc = 0.0;
    for (int t = 0; t < num_tiles; t++) {
        A_tile[y * tile_dim + x] = A[(tile_row * tile_dim + y) * N + t * tile_dim + x];
        B_tile[y * tile_dim + x] = B[(t * tile_dim + y) * N + tile_col * tile_dim + x];
        __syncthreads();
        for (int k = 0; k < tile_dim; k++) {
            acc += A_tile[y * tile_dim + k] * B_tile[tile_dim * k + x];
        }
        __syncthreads();
    }
    C[(tile_row * tile_dim + y) * N + tile_col * tile_dim + x] = acc;

}

int main() {
    // A @ B = C
    // A is a NxN matrix, B is a NxN matrix, C is a NxN matrix
    int N = 2048; 

    // Allocate device memory
    float *A_d, *B_d, *C_d, *C_ref;
    CUDACHECK(cudaMalloc(&A_d, sizeof(float) * N * N));
    CUDACHECK(cudaMalloc(&B_d, sizeof(float) * N * N));
    CUDACHECK(cudaMalloc(&C_d, sizeof(float) * N * N));
    CUDACHECK(cudaMalloc(&C_ref, sizeof(float) * N * N));

    // Initialize matrices on device
    fill_random(A_d, N * N, 2024);
    fill_random(B_d, N * N, 2025);
    CUDACHECK(cudaDeviceSynchronize());

    // Compute reference GEMM on device
    reference_gemm(C_ref, A_d, B_d, N);
    CUDACHECK(cudaDeviceSynchronize());

    dim3 grid(N / NUM_THREADS, N / NUM_THREADS);
    dim3 block(NUM_THREADS, NUM_THREADS);

    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start));
    kernel<<<grid, block>>>(A_d, B_d, C_d, N);
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    // duration and TFLOPs
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double microseconds = milliseconds * 1000.0;
    double flops = double(2.0) * N * N * N;
    double tflops = (flops / microseconds) / 1e6;
    std::cout << "kernel execution time: " << microseconds << " us\n";
    std::cout << "performance: " << tflops << " TFLOPs\n";

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