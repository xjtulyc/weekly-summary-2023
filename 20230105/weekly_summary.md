# 周报-20230105

## 1. 项目进展Ultrasound_vid

### 1.1. 已完成

· 使用最新的配置文件跑baseline
```
--config-file configs/RDN-LSTM/BUS_BasicConfig_20221108_hardmining_fp_thresh0.6_by_video_rate0.7_fold0_iter_10w.yaml
```
### 1.2. 下周任务

· 以[0.3, 0.4, 0.5, 0.6, 0.7]的概率随机选择静止帧训练模型，并比较性能

## 2. 论文阅读

### 2.1. Multi-Granularity Cross-modal Alignment for Generalized Medical Visual Representation Learning

- ``DoI`` https://arxiv.org/abs/2210.06044
#### ``2.1.1. 概述``

使用CLIP结构，学习和医学图像与诊断报告的共同表征。在病理区域级、实例级和疾病级三个层面，通过损失函数实现语义对齐。

