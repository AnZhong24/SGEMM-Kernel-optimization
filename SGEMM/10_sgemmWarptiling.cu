#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int WARPSIZE = 32; 
 
   const int BN = 128;
  const  int BM = 64;
  const int BK = 8;
  const int WN = 64;
  const int WM = 32;
  const int WNITER = 2;
  const int TN = 4;
  const int TM = 4;

  
const int NUM_THREADS = 128;

__global__ void __launch_bounds__(NUM_THREADS)
    sgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C) {

  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  // Placement of the warp in the threadblock tile
  const int warpIdx = threadIdx.x / WARPSIZE; 
  const int warpCol = warpIdx % (BN / WN);
  const int warpRow = warpIdx / (BN / WN);

  // size of the warp subtile
  constexpr int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr int WSUBM = WM / WMITER; // 64/2=32
  constexpr int WSUBN = WN / WNITER; // 32/2=16


  const int threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]
  const int threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
  const int threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4


  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];


  A += cRow * BM * K;
  B += cCol * BN;

  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;



  const int innerRowA = threadIdx.x / (BK / 4);
  const int innerColA = threadIdx.x % (BK / 4);
  constexpr int rowStrideA = (NUM_THREADS * 4) / BK;
  const int innerRowB = threadIdx.x / (BN / 4);
  const int innerColB = threadIdx.x % (BN / 4);
  constexpr int rowStrideB = NUM_THREADS / (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  // we cache into registers on the warptile level
  float regM[WMITER * TM] = {0.0};
  float regN[WNITER * TN] = {0.0};

  // outer-most loop over block tiles
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    for (int offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
      float4 tmp = reinterpret_cast<float4 *>(
          &A[(innerRowA + offset) * K + innerColA * 4])[0];
      // transpose A while storing it
      As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
      As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
      As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
      As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }

    for (int offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
      reinterpret_cast<float4 *>(
          &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
          reinterpret_cast<float4 *>(
              &B[(innerRowB + offset) * N + innerColB * 4])[0];
    }
    __syncthreads();

    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // populate registers for whole warptile
      for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (int i = 0; i < TM; ++i) {
          regM[wSubRowIdx * TM + i] =
              As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
                 threadRowInWarp * TM + i];
        }
      }
      for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        for (int i = 0; i < TN; ++i) {
          regN[wSubColIdx * TN + i] =
              Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
                 threadColInWarp * TN + i];
        }
      }

      // execute warptile matmul
      for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
          // calculate per-thread results
          for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
              threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                            (wSubColIdx * TN) + resIdxN] +=
                  regM[wSubRowIdx * TM + resIdxM] *
                  regN[wSubColIdx * TN + resIdxN];
            }
          }
        }
      }
    }
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
    __syncthreads();
  }

  // write out the results
  for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // move C pointer to current warp subtile
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (int resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // load C vector into registers
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0];
          // perform GEMM update in reg
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          // write back
          reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0] = tmp;
        }
      }
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
   
  dim3 blockDim( NUM_THREADS);  
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  sgemmWarptiling<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
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
