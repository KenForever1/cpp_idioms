#include <torch/extension.h>

void softmax_exp_forward_cuda(const torch::Tensor &x, torch::Tensor &tmp);
void softmax_div_forward_cuda(const torch::Tensor &tmp, const torch::Tensor &sum_val, torch::Tensor &output);

torch::Tensor softmax_forward(torch::Tensor x) {
  auto tmp = torch::empty_like(x);
  softmax_exp_forward_cuda(x, tmp);
  auto sum_val = tmp.sum();
  auto output = torch::empty_like(x);
  softmax_div_forward_cuda(tmp, sum_val, output);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax_forward", &softmax_forward, "Softmax forward (CUDA)");
}