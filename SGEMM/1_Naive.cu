#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
  const  int x = blockIdx.x * blockDim.x + threadIdx.x;
  const  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

void initialize_matrix(float* mat, int rows, int cols) {
  for (int i = 0; i < rows * cols; ++i) {
    mat[i] = static_cast<float>(rand()) / RAND_MAX;
  }
}

int main() {
  int M = 4092;
  int N = 4092;
  int K = 4092;

  float alpha = 1.0f;
  float beta = 0.0f;

  size_t size_A = M * K * sizeof(float);
  size_t size_B = K * N * sizeof(float);
  size_t size_C = M * N * sizeof(float);

  float *h_A = (float*)malloc(size_A);
  float *h_B = (float*)malloc(size_B);
  float *h_C = (float*)malloc(size_C);

  initialize_matrix(h_A, M, K);
  initialize_matrix(h_B, K, N);
  initialize_matrix(h_C, M, N);

  float *d_A, *d_B, *d_C;
  cudaMalloc((void**)&d_A, size_A);
  cudaMalloc((void**)&d_B, size_B);
  cudaMalloc((void**)&d_C, size_C);

  cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

  dim3 blockDim(32, 32, 1);
  dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);

  float gflops_naive = (2.0f * M * N * K) / (elapsedTime / 1000.0f) / 1e9;

  cublasHandle_t handle;
  cublasCreate(&handle);

  cudaEventRecord(start, 0);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);

  float elapsedTime_cublas;
  cudaEventElapsedTime(&elapsedTime_cublas, start, stop);

  float gflops_cublas = (2.0f * M * N * K) / (elapsedTime_cublas / 1000.0f) / 1e9;

  cublasDestroy(handle);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  free(h_A);
  free(h_B);
  free(h_C);

  printf("Naive GFLOPs: %f\n", gflops_naive);
  printf("cuBLAS GFLOPs: %f\n", gflops_cublas);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
