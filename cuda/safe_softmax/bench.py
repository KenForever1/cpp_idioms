import torch
from torch.utils.cpp_extension import load
import time


cuda_ext = load(
    name="safe_softmax_cuda",
    sources=["safe_softmax_cuda.cpp", "safe_softmax_cuda_kernel.cu"],
    verbose=True
)

# 测试验证
X = torch.randn(1000, 10000, device="cuda")  # 大矩阵测试

start = time.time()
cuda_result = cuda_ext.safe_softmax_cuda(X)
torch.cuda.synchronize()
custom_time = time.time() - start

# PyTorch原生实现
start = time.time()
native_result = torch.nn.functional.softmax(X, dim=-1)
torch.cuda.synchronize()  # 确保CUDA操作完成
native_time = time.time() - start

print("最大误差:", torch.max(torch.abs(cuda_result - native_result)))  # 应接近0
print(f"原生耗时: {native_time:.5f}s | 扩展耗时: {custom_time:.5f}s")
