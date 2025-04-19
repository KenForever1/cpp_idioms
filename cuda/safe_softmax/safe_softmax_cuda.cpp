#include <torch/extension.h>



torch::Tensor safe_softmax_cuda(torch::Tensor X);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("safe_softmax_cuda", &safe_softmax_cuda, "CUDA Safe Softmax");
}
