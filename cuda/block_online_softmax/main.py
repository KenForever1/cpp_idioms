import torch
from torch.utils.cpp_extension import load

# 加载CUDA扩展
softmax_forward = load(
    name='block_online_softmax',
    sources=['main.cpp', 'softmax.cu'],
    extra_cuda_cflags=['-O2']
)

# 创建输入张量并移动到CUDA设备
X = torch.tensor([[-0.3, 0.2, 0.5, 0.7, 0.1, 0.8]], device='cuda')


# 手动计算以验证结果
X_exp = torch.exp(X)
X_exp_sum = X_exp.sum()
X_softmax_hand = X_exp / X_exp_sum

print("手动计算Softmax:", X_softmax_hand)

# CUDA实现
cuda_result = softmax_forward.block_online_softmax(X, 3)

# Python参考实现
X_block = torch.split(X, split_size_or_sections=3, dim=1)
X_block_0_max = X_block[0].max()
X_block_0_sum = torch.exp(X_block[0] - X_block_0_max).sum()
X_block_1_max = X_block[1].max()
X_block_1_sum = torch.exp(X_block[1] - X_block_1_max).sum()
X_max_update = torch.max(X_block_0_max, X_block_1_max)
X_sum_update = X_block_0_sum * torch.exp(X_block_0_max - X_max_update) + \
                torch.exp(X_block[1] - X_max_update).sum()
ref_result = torch.exp(X - X_max_update) / X_sum_update

print("CUDA结果:", cuda_result.cpu())
print("参考结果:", ref_result.cpu())
print("最大误差:", torch.max(torch.abs(cuda_result - ref_result)).item())

