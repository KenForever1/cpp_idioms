## 实现
+ CUDA核函数‌

exp_kernel：计算输入张量的指数并存储到临时张量。
div_kernel：将临时张量的每个元素除以总和，得到最终Softmax结果。

+ PyTorch集成‌（main.py）
+ 
使用torch.utils.cpp_extension.load将C++/CUDA代码编译为Python模块。


## 指定CUDA架构

编译时指定特定的CUDA架构可以减少编译时间。默认情况下，库可能会为所有支持的CUDA架构生成二进制代码，这可能需要相当长的时间。如果你只指定自己实际使用的那些架构，就可以节省大量时间。也可以减少编译产物占用的磁盘空间。

```bash
python -c "import torch; print(torch.cuda.get_device_capability())"
>>> (8, 9) # 对应的就是 8.9

TORCH_CUDA_ARCH_LIST="8.9" python main.py
```

执行结果：
```bash
手动计算Softmax: tensor([0.0827, 0.1364, 0.1841, 0.2249, 0.1234, 0.2485], device='cuda:0')
CUDA计算Softmax: tensor([0.0827, 0.1364, 0.1841, 0.2249, 0.1234, 0.2485], device='cuda:0')
```

## 错误解决

+ [pytorch env version `GLIBCXX_3.4.32' not found](https://zhuanlan.zhihu.com/p/28103840741), 将系统默认的libstdc++.so.6软链接到pytorch的libstdc++.so.6