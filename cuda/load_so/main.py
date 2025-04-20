
import os
from torch.utils.cpp_extension import load, _import_module_from_library
import torch
# try:
#     user_home_path = os.path.expanduser('~')
#     fused = _import_module_from_library('fused', user_home_path+'/.cache/torch_extensions/fused', True)
#     print(f'Load fused from {user_home_path}/.cache/torch_extensions/fused')
# except:
#     module_path = os.path.dirname(__file__)
#     fused = load(
#         'fused',
#         sources=[
#             os.path.join(module_path, 'fused_bias_act.cpp'),
#             os.path.join(module_path, 'fused_bias_act_kernel.cu'),
#         ],
#     )
#     print(f'Build fused from cpp & cu files')

user_home_path = os.path.expanduser('~')


block_online_softmax = _import_module_from_library('block_online_softmax', user_home_path+'/.cache/torch_extensions/py310_cu121/block_online_softmax', True)
print(f'Load softmax from {user_home_path}/.cache/torch_extensions/py310_cu121/block_online_softmax')

online_softmax = _import_module_from_library('online_softmax', user_home_path+'/.cache/torch_extensions/py310_cu121/online_softmax', True)
print(f'Load fused from {user_home_path}/.cache/torch_extensions/py310_cu121/online_softmax')


safe_softmax_cuda = _import_module_from_library('safe_softmax_cuda', user_home_path+'/.cache/torch_extensions/py310_cu121/safe_softmax_cuda', True)
print(f'Load softmax from {user_home_path}/.cache/torch_extensions/py310_cu121/safe_softmax_cuda')


# 创建输入张量并移动到CUDA设备
X = torch.tensor([[-0.3, 0.2, 0.5, 0.7, 0.1, 0.8] * 100], device='cuda')

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    X_online_softmax_cuda = online_softmax.online_softmax(X)
    # print("CUDA计算online Softmax:", X_softmax_cuda)
# Self CPU time total: 11.053ms
# Self CUDA time total: 10.688ms

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=6))


with torch.autograd.profiler.profile(use_cuda=True) as prof:
    X_block_online_softmax_cuda = block_online_softmax.block_online_softmax(X, 6)
    # print("CUDA计算online Softmax:", X_block_online_softmax_cuda)

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))


with torch.autograd.profiler.profile(use_cuda=True) as prof:
    X_softmax_cuda = safe_softmax_cuda.safe_softmax_cuda(X)
    # print("CUDA计算Softmax:", X_softmax_cuda)
# Self CPU time total: 497.000us
# Self CUDA time total: 45.000us
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))


with torch.autograd.profiler.profile(use_cuda=True) as prof:
    # 手动计算以验证结果
    X_exp = torch.exp(X)
    X_exp_sum = X_exp.sum()
    X_softmax_hand = X_exp / X_exp_sum
    # print("手动计算Softmax:", X_softmax_hand)
# Self CPU time total: 18.473ms
# Self CUDA time total: 18.473ms
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('softmax values sanity check:', torch.allclose(X_online_softmax_cuda, X_softmax_hand, rtol=0, atol=1e-02))
