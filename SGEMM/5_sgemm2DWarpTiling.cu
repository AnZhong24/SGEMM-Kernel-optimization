#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) ((M) + (N)-1) / (N)
 const int BM = 128;
  const int BN = 128;
  const int BK = 8;
  const int TM = 8;
  const int TN = 8;
__global__ void sgemm2DWarpTiling(int M, int N, int K, float alpha,
                                  const float *A, const float *B, float beta,
                                  float *C) {
 
  const  int cRow = blockIdx.y;
  const  int cCol = blockIdx.x;

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


  const  int innerRowA = threadIdx.x / BK;
  const  int innerColA = threadIdx.x % BK;


  const  int strideA = numThreadsBlocktile / BK;
  const  int innerRowB = threadIdx.x / BN;
  const  int innerColB = threadIdx.x % BN;
  const  int strideB = numThreadsBlocktile / BN;


  float threadResults[TM * TN] = {0.0};

  float regM[TM] = {0.0};
  float regN[TN] = {0.0};


  for ( int bkIdx = 0; bkIdx < K; bkIdx += BK) {

    for ( int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for ( int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    

    A += BK;      
    B += BK * N;  


    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
     

      for (int i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (int i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for ( int resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }


  for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          alpha * threadResults[resIdxM * TN + resIdxN] +
          beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
    }
  }
}


 