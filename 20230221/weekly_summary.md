# 周报-20230221

## 1. Ultrasound VID

### 1.1. US 数据集

Before MICCAI (3.9):
1. 把所有的BUS/TUS数据集统一为yizhun数据格式；
2. 继续寻找US数据集；
3. 对于分类数据集使用SOTA模型来inference，获得伪标签；（暂定） 

yizhun 400k $$

### 1.2. MICCAI 23 Related Work (ddl: 2.22)

尽量全
1. ultrasound 2D image detection
2. Optical Flow Method in ultrasound

### 1.2. 在
## 一些记录

1. 查看pytorch运行时真正调用的cuda版本

```python
import torch
import torch.utils
import torch.utils.cpp_extension

torch.utils.cpp_extension.CUDA_HOME        #输出 Pytorch 运行时使用的 cuda 
```