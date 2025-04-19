#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>


__global__ void online_update_kernel(
  const float* input,    // 输入矩阵 [batch_size, seq_len]
  float* running_max,    // 运行中最大值 [batch_size]
  float* running_sum,    // 运行中指数和 [batch_size]
  int seq_len            // 当前序列长度
) {
  int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_idx >= gridDim.x) return;

  float current_max = -INFINITY;
  float current_sum = 0.0f;

  // 增量处理每个时间步
  for (int pos = 0; pos < seq_len; ++pos) {
      float x = input[batch_idx * seq_len + pos];
      
      // 计算新最大值
      float new_max = fmaxf(current_max, x);
      
      // 调整历史指数和
      float exp_ratio = expf(current_max - new_max);
      float adjusted_sum = current_sum * exp_ratio;
      
      // 添加新项
      float new_term = expf(x - new_max);
      current_sum = adjusted_sum + new_term;
      current_max = new_max;
  }

  // 写入全局内存
  running_max[batch_idx] = current_max;
  running_sum[batch_idx] = current_sum;
}

__global__ void compute_softmax_kernel(
  const float* input,
  const float* running_max,
  const float* running_sum,
  float* output,
  int seq_len
) {
  int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_idx >= gridDim.x) return;

  float max_val = running_max[batch_idx];
  float sum_val = running_sum[batch_idx];

  for (int pos = 0; pos < seq_len; ++pos) {
      float x = input[batch_idx * seq_len + pos];
      output[batch_idx * seq_len + pos] = expf(x - max_val) / sum_val;
  }
}

torch::Tensor online_softmax_cuda(torch::Tensor X) {
    auto batch_size = X.size(0);
    auto seq_len = X.size(1);
    auto options = X.options();

    // 初始化运行状态
    auto running_max = torch::full({batch_size}, -INFINITY, options);
    auto running_sum = torch::zeros({batch_size}, options);

    // 配置CUDA执行参数
    const int threads_per_block = 256;
    const int blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    // 阶段1：增量更新最大值和指数和
    online_update_kernel<<<blocks, threads_per_block>>>(
        X.data_ptr<float>(),
        running_max.data_ptr<float>(),
        running_sum.data_ptr<float>(),
        seq_len
    );

    // 阶段2：统一计算Softmax
    auto output = torch::empty_like(X);
    compute_softmax_kernel<<<blocks, threads_per_block>>>(
        X.data_ptr<float>(),
        running_max.data_ptr<float>(),
        running_sum.data_ptr<float>(),
        output.data_ptr<float>(),
        seq_len
    );

    return output;
}

