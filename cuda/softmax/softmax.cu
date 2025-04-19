#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void exp_kernel(const float *x, float *tmp, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    tmp[idx] = expf(x[idx]);
  }
}

__global__ void div_kernel(const float *tmp, const float *sum_val, float *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = tmp[idx] / (*sum_val);
  }
}

void softmax_exp_forward_cuda(const torch::Tensor &x, torch::Tensor &tmp) {
  const int n = x.numel();
  const float *x_data = x.data_ptr<float>();
  float *tmp_data = tmp.data_ptr<float>();

  const int block_size = 256;
  const int grid_size = (n + block_size - 1) / block_size;

  exp_kernel<<<grid_size, block_size>>>(x_data, tmp_data, n);
}

void softmax_div_forward_cuda(const torch::Tensor &tmp, const torch::Tensor &sum_val, torch::Tensor &output) {
  const int n = tmp.numel();
  const float *tmp_data = tmp.data_ptr<float>();
  const float *sum_val_data = sum_val.data_ptr<float>();
  float *output_data = output.data_ptr<float>();

  const int block_size = 256;
  const int grid_size = (n + block_size - 1) / block_size;

  div_kernel<<<grid_size, block_size>>>(tmp_data, sum_val_data, output_data, n);
}