online softmax

```python
X_pre = X[:-1]
print('input x')
print(X)
print(X_pre)
print(X[-1])

# we calculative t-1 time Online Softmax
X_max_pre = X_pre.max()
X_sum_pre = torch.exp(X_pre - X_max_pre).sum()

# we calculative t time Online Softmax
X_max_cur = torch.max(X_max_pre, X[-1]) # X[-1] is new data
X_sum_cur = X_sum_pre * torch.exp(X_max_pre - X_max_cur) + torch.exp(X[-1] - X_max_cur)

# final we calculative online softmax
X_online_softmax = torch.exp(X - X_max_cur) / X_sum_cur
print('online softmax result: ', X_online_softmax)
```