# 周报-20230202


## 1. 项目进展Ultrasound_vid_static_frame

### 1.1. 对于``非静止帧``视频和``人工采样（随机在有标注框的位置造静止帧）``视频统计非静止帧和静止帧的结果

#### 1.1.1. 20230112实验结果分析

``在不同split来验证是否有效``

### 1.2. 试一下跑甲状腺的数据，看看会不会提点，就按照现有版本改，然后跑甲状腺的数据

#### 1.1.2. 俞在甲状腺上的实验

0.4的概率剔除静止帧在甲状腺上FP反而变差了1.1/min=>1.3/min。移除概率并不是0.4都好的。

## 论文阅读

### 1.2. You Only Look Once: Unified, Real-Time Object Detection

Author: oseph Redmon∗, Santosh Divvala∗†, Ross Girshick¶, Ali Farhadi∗†

