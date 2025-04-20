#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void block_stats_kernel(
  const float* input,
  float* block_max,
  float* block_sum_exp,
  int seq_len,
  int block_size,
  int num_blocks
) {
  int bid = blockIdx.x;  // 批次索引
  int block_id = blockIdx.y; // 块索引
  int tid = threadIdx.x; // 线程索引

  __shared__ float shared_max[256];
  __shared__ float shared_sum[256];
  shared_max[tid] = -INFINITY;
  shared_sum[tid] = 0.0f;

  // 计算当前块范围
  int block_start = block_id * block_size;
  int block_end = min(block_start + block_size, seq_len);
  int elements = block_end - block_start;

  // 阶段1: 计算块内最大值
  for (int i = block_start + tid; i < block_end; i += blockDim.x) {
      float val = input[bid * seq_len + i];
      shared_max[tid] = fmaxf(shared_max[tid], val);
  }
  __syncthreads();

  // 归约求块最大值
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride && tid + stride < elements) {
          shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
      }
      __syncthreads();
  }

  // 阶段2: 计算块内指数和
  float current_max = shared_max[0];
  for (int i = block_start + tid; i < block_end; i += blockDim.x) {
      float val = input[bid * seq_len + i];
      shared_sum[tid] += expf(val - current_max);
  }
  __syncthreads();

  // 归约求块指数和
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride && tid + stride < elements) {
          shared_sum[tid] += shared_sum[tid + stride];
      }
      __syncthreads();
  }

  // 写入全局内存
  if (tid == 0) {
      block_max[bid * num_blocks + block_id] = current_max;
      block_sum_exp[bid * num_blocks + block_id] = shared_sum[0];
  }
}


__global__ void merge_blocks_kernel(
  const float* block_max,
  const float* block_sum_exp,
  float* global_max,
  float* global_sum_exp,
  int num_blocks
) {
  int bid = blockIdx.x; // 批次索引

  // 初始化
  float current_max = -INFINITY;
  float current_sum = 0.0f;

  // 顺序合并各块统计量
  for (int i = 0; i < num_blocks; ++i) {
      float block_max_val = block_max[bid * num_blocks + i];
      float block_sum_val = block_sum_exp[bid * num_blocks + i];

      float new_max = fmaxf(current_max, block_max_val);
      float exp_ratio = expf(current_max - new_max);
      
      current_sum = current_sum * exp_ratio + 
                   block_sum_val * expf(block_max_val - new_max);
      current_max = new_max;
  }

  // 写入最终结果
  global_max[bid] = current_max;
  global_sum_exp[bid] = current_sum;
}

torch::Tensor block_online_softmax_cuda(
  torch::Tensor X,
  int block_size=3  // 默认块大小
) {

  // cudaDeviceProp prop;
  // cudaGetDeviceProperties(&prop, 0);
  // int max_grid_x = *prop.maxGridSize; 

  // printf("max_grid_x=%d\n", max_grid_x);

  auto batch_size = X.size(0);
  auto seq_len = X.size(1);
  auto options = X.options();

  // 计算分块数
  int num_blocks = (seq_len + block_size - 1) / block_size;

  // 中间结果缓存
  auto block_max = torch::zeros({batch_size, num_blocks}, options);
  auto block_sum_exp = torch::zeros_like(block_max);

  // 阶段1: 并行计算各块统计量
  dim3 grid(batch_size, num_blocks);
  dim3 block(256);
  block_stats_kernel<<<grid, block>>>(
      X.data_ptr<float>(),
      block_max.data_ptr<float>(),
      block_sum_exp.data_ptr<float>(),
      seq_len,
      block_size,
      num_blocks
  );

  // 阶段2: 合并块统计量
  auto global_max = torch::zeros({batch_size}, options);
  auto global_sum_exp = torch::zeros({batch_size}, options);
  
  merge_blocks_kernel<<<batch_size, 1>>>(
      block_max.data_ptr<float>(),
      block_sum_exp.data_ptr<float>(),
      global_max.data_ptr<float>(),
      global_sum_exp.data_ptr<float>(),
      num_blocks
  );

  // 阶段3: 计算最终softmax
  auto X_max = global_max.unsqueeze(1).expand({-1, seq_len});
  auto X_sum = global_sum_exp.unsqueeze(1).expand({-1, seq_len});
  return torch::exp(X - X_max) / X_sum;
}
