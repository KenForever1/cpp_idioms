#include <torch/extension.h>

torch::Tensor block_online_softmax_cuda(
  torch::Tensor X,
  int block_size=3  // 默认块大小
);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("block_online_softmax", &block_online_softmax_cuda, "Block Online Softmax");
}
