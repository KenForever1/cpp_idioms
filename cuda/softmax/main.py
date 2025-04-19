import torch
from torch.utils.cpp_extension import load

# 加载CUDA扩展
softmax_forward = load(
    name='softmax_forward',
    sources=['main.cpp', 'softmax.cu'],
    extra_cuda_cflags=['-O2']
)

X = torch.tensor([-0.3, 0.2, 0.5, 0.7, 0.1, 0.8], device='cuda')

X_softmax_cuda = softmax_forward.softmax_forward(X)

# 手动计算以验证结果
X_exp = torch.exp(X)
X_exp_sum = X_exp.sum()
X_softmax_hand = X_exp / X_exp_sum

print("手动计算Softmax:", X_softmax_hand)
print("CUDA计算Softmax:", X_softmax_cuda)
