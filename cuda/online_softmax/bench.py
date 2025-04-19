import torch
from torch.utils.cpp_extension import load
import numpy as np
import math

cuda_ext = load(
    name="online_softmax",
    sources=["main.cpp", "softmax.cu"],
    verbose=True
)

X = torch.tensor([[1.0, 3.0, 5.0]], device="cuda")
result = cuda_ext.online_softmax(X)
print("Result:\n", result.cpu().numpy())

# 验证
max_val = 5.0
sum_exp = math.exp(1-5) + math.exp(3-5) + math.exp(5-5)
expected = torch.tensor([[math.exp(1-5)/sum_exp, 
                         math.exp(3-5)/sum_exp,
                         math.exp(5-5)/sum_exp]])
print("Expected:\n", expected.numpy())
print("最大误差:", torch.max(torch.abs(result - expected.to("cuda"))).item())
