# 周报-20230119

## 1. 项目进展Ultrasound_vid_static_frame

### 1.1. 对于``非静止帧``视频和``人工采样（随机在有标注框的位置造静止帧）``视频统计非静止帧和静止帧的结果


#### 1.1.1. 20230112实验结果分析

``在不同split来验证是否有效``

根据上周的实验结果，如下图所示。

![实验结果](file\experiment_20230112.png)

以0.4概率移除静止帧在大多数指标更好。初步认为是训练的时候给的样例更加明确（有bbox，而且展示的更慢），效果会更好。即加入了简单样本后会提升，但是简单样本不是越多越好。

之前db师兄做的实验结论

```
1. 训练rm_static_rate=1.0，测试rm_static_rate=1.0
2. 训练rm_static_rate=0.0，测试rm_static_rate=1.0
1.比2.好
```

#### 1.1.2. 为什么推理的时候需要剔除静止帧

因为静止帧相对比较简单，如果加入静止帧一起测试的话点数很高，看不出啥东西。但是现在发现静止帧如果完全不加也会有一些问题。

#### 1.1.3. 方法

在tests/test_evaluator.py里面可以测试prediction。只需要给路径就行了，重写evaluator。

跑之前需要提交一个分支作代码检查。

## 2. 项目进展Ultrasound_vid_window_size

### 2.1. 文献调研

在video object segmentation和video object detection两个任务的sota榜单找使用memory管理的方法。

[video segmentation](https://paperswithcode.com/task/video-segmentation)

[video semantic segmentation](https://paperswithcode.com/task/video-semantic-segmentation/codeless)

[video object segmentation](https://paperswithcode.com/task/video-object-segmentation)

[video panoptic segmentation](https://paperswithcode.com/paper/video-panoptic-segmentation-1)

[video instance segmentation](https://paperswithcode.com/task/video-instance-segmentation)

### 2.2. 使用了memory的工作

|题目|期刊/会议|任务|机制|
|:---|:---:|:---:|---:|
|[XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model](https://paperswithcode.com/paper/xmem-long-term-video-object-segmentation-with)|ECCV22|video object segmentation|使用Atkinson-Shiffrin Memory Model，将memory分割为感知记忆、工作记忆和长期记忆三种模块|

## 2. 论文阅读

### 2.1. CLIP-Driven Universal Model for Organ Segmentation and Tumor Detection

## 2.2. MICCAI22 Part III Colonoscopy部分

[MICCAI 2022 Part III Colonoscopy](https://github.com/xjtulyc/MICCAI2022_paper_reading/blob/main/Part%20III/notes.md)
