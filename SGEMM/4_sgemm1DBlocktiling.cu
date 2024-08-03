#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define BM 64
#define BN 64
#define BK 8
#define TM 8


__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  const  int cRow = blockIdx.y;
  const  int cCol = blockIdx.x;

  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  const  int innerColA = threadIdx.x % BK;
  const  int innerRowA = threadIdx.x / BK;
  const  int innerColB = threadIdx.x % BN;
  const  int innerRowB = threadIdx.x / BN;

  float threadResults[TM] = {0.0};

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    A += BK;
    B += BK * N;

    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float tmpB = Bs[dotIdx *  BN + threadCol];
      for (int resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  for (int resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta * C[(threadRow * TM + resIdx) * N + threadCol];
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

  dim3 blockDim(BM * BK);  
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM)); 

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  sgemm1DBlocktiling<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);

  float gflops = (2.0f * M * N * K) / (elapsedTime / 1000.0f) / 1e9;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  free(h_A);
  free(h_B);
  free(h_C);

  printf("GFLOPs: %f\n", gflops);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
