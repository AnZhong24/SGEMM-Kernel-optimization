#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "params.h"
#define CEIL_DIV(M, N) ((M) + (N)-1) / (N)

  
__global__ void sgemmVectorize(int M, int N, int K, float alpha, float *A,
                               float *B, float beta, float *C) {

  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  const int totalResultsBlocktile = BM * BN;
  
  const int numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  
  assert(numThreadsBlocktile == blockDim.x);

  
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  
  
  const int innerRowA = threadIdx.x / (BK / 4);
  const int innerColA = threadIdx.x % (BK / 4);
  
  
  const int rowStrideA = (numThreadsBlocktile * 4) / BK;
  const int innerRowB = threadIdx.x / (BN / 4);
  const int innerColB = threadIdx.x % (BN / 4);
  
  
  
  const int rowStrideB = numThreadsBlocktile / (BN / 4);

  
  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    
	
	float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    /*
     * Writing it as an unrolled loop has worse performance, because it doesn't
     * guarantee alignment of the loads
     * Bs[innerRowB * BN + innerColB * 4 + 0] =
     *    B[innerRowB * N + innerColB * 4 + 0];
     * Bs[innerRowB * BN + innerColB * 4 + 1] =
     *    B[innerRowB * N + innerColB * 4 + 1];
     * Bs[innerRowB * BN + innerColB * 4 + 2] =
     *    B[innerRowB * N + innerColB * 4 + 2];
     * Bs[innerRowB * BN + innerColB * 4 + 3] =
     *    B[innerRowB * N + innerColB * 4 + 3];
     */

    reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    __syncthreads();

    
	A += BK;     
    B += BK * N;  
    
	for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      
	  for (int i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      }
      for (int i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  
  for (int resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      
	  float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      
	  tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
      tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
      tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
      tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
      
	  reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
          tmp;
    }
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
   
  dim3 blockDim( (BM*BN)/(TM*TN));  
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  sgemmVectorize<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
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
