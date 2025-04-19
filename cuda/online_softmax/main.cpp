#include <torch/extension.h>

torch::Tensor online_softmax_cuda(torch::Tensor X);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("online_softmax", &online_softmax_cuda, "CUDA Online Softmax");
}
