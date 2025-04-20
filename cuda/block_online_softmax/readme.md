block online softmax

```python
X_block = torch.split(X, split_size_or_sections = 3 , dim = 0) 
print(X)
print(X_block)

# we parallel calculate  different block max & sum
X_block_0_max = X_block[0].max()
X_block_0_sum = torch.exp(X_block[0] - X_block_0_max).sum()

X_block_1_max = X_block[1].max()
X_block_1_sum = torch.exp(X_block[1] - X_block_1_max).sum()

# online block update max & sum
X_block_1_max_update = torch.max(X_block_0_max, X_block_1_max) # X[-1] is new data
X_block_1_sum_update = X_block_0_sum * torch.exp(X_block_0_max - X_block_1_max_update) \
                     + torch.exp(X_block[1] - X_block_1_max_update).sum() # block sum

X_block_online_softmax = torch.exp(X - X_block_1_max_update) / X_block_1_sum_update
print(X_block_online_softmax)
```

 Flash Attention所谓的memory-IO-effiecent，实际上受益于块状的softmax，加载部分QK到缓存计算局部的注意力分数。