#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

/**
 * 每个线程块处理一行数据（row = blockIdx.x），共享内存存储临时最大值‌
 * 使用树形归约（tree reduction）优化并行最大值计算‌
 */
__global__ void reduce_max_kernel(float* input, float* max_values, int rows, int cols) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int row = blockIdx.x;

  // 加载数据到共享内存
  sdata[tid] = -INFINITY;
  for (int i = tid; i < cols; i += blockDim.x) {
      sdata[tid] = fmaxf(sdata[tid], input[row * cols + i]);
  }
  __syncthreads();

  // 块内归约求最大值
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) {
          sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
      }
      __syncthreads();
  }

  if (tid == 0) {
      max_values[row] = *sdata;
    //   printf("max_values[%d] = %f\n", row, max_values[row]);
  }
}


__global__ void softmax_kernel(float* input, float* output, float* max_values, int rows, int cols) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = blockIdx.y;

  if (col < cols && row < rows) {
      // 减去行最大值并计算指数
      float val = expf(input[row * cols + col] - max_values[row]);
      output[row * cols + col] = val;

      // 共享内存求和
      __shared__ float sum_shared;
      if (threadIdx.x == 0) {
          sum_shared = 0.0f;
      }
      __syncthreads();

      atomicAdd(&sum_shared, val);
      __syncthreads();

      // 归一化
      output[row * cols + col] /= sum_shared;
  }
}

torch::Tensor safe_softmax_cuda(torch::Tensor X) {
    auto max_values = torch::empty(X.size(0), X.options());
    auto output = torch::empty_like(X);

    int rows = X.size(0);
    int cols = X.size(1);

    // 调用最大值归约核函数
    dim3 grid_max(rows);
    dim3 block_max(256);
    reduce_max_kernel<<<grid_max, block_max, sizeof(float)*256>>>(
        X.data_ptr<float>(), 
        max_values.data_ptr<float>(), 
        rows, cols
    );

    // 调用Softmax核函数
    dim3 grid_softmax((cols + 255) / 256, rows);
    dim3 block_softmax(256);
    softmax_kernel<<<grid_softmax, block_softmax>>>(
        X.data_ptr<float>(), 
        output.data_ptr<float>(), 
        max_values.data_ptr<float>(), 
        rows, cols
    );

    return output;
}