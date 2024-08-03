
# SGEMM-Kernel-optimization
# CUDA Matrix Multiplication Kernel Optimization

This repository contains various CUDA kernel implementations for matrix multiplication, with performance comparisons to the cuBLAS library. The implementation is based on the tutorial: [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com).

## Tested Environment

- **GPU:** NVIDIA GTX 1090

## Performance Comparison

The following table shows the performance of different kernel optimizations tested on a GTX 1090 GPU, compared to the cuBLAS implementation:

| Kernel              | GFLOPs/s | Performance Relative to cuBLAS |
|---------------------|----------|--------------------------------|
| Naive               | 178.1    | 6.15%                          |
| GMEM Coalescing     | 491.5    | 16.97%                         |
| SMEM Caching        | 491.5    | 16.97%                         |
| 1D Blocktiling      | 938.7    | 32.41%                         |
| 2D Blocktiling      | 1785.1   | 61.63%                         |
| Vectorized Mem Access | 1939.5 | 66.96%                         |
| Autotuning          | 2361.8   | 81.54%                         |
| Warptiling          | 2510.5   | 86.68%                         |
| cuBLAS              | 2896.4   | 100.00%                        |


