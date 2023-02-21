# 周报-20230221

1. 毕业设计开题答辩
2. 训练了BUS_rms_0.4和TUS_rms_0.2（还未进行eval，今晚统计结果）
3. Pycharm学生证书到期无法续费，转用vsc

## 1. Ultrasound VID

### 1.1. US 数据集

Before MICCAI (3.9):
1. 把所有的BUS/TUS数据集统一为yizhun数据格式；
2. 继续寻找US数据集；
3. 对于分类数据集使用SOTA模型来inference，获得伪标签；（暂定） 

### 1.2. MICCAI 23 Related Work (ddl: 2.22)

尽量全
1. ultrasound 2D image detection
2. Optical Flow Method in ultrasound

### 1.3. 在甲状腺数据集上训练0.2, 0.4, 0.6, 0.8, 1.0移除静止帧的检测模型的表现。


## 2. 一些记录

1. 查看pytorch运行时真正调用的cuda版本

```python
import torch
import torch.utils
import torch.utils.cpp_extension

torch.utils.cpp_extension.CUDA_HOME        #输出 Pytorch 运行时使用的 cuda 
```

2. 服务器使用

[集群使用说明](file/集群使用说明.pdf)

[VPN](file/windows使用vpn教程.docx)（目前gitlab也需要VPN）

3. CMD上传文件到服务器指定路径

```shell
scp -r {file} username@ip:{filepath}
```

