# 周报-20230119

## 1. 项目进展Ultrasound_vid_static_frame

### 1.1. 对于``非静止帧``视频和``人工采样（随机在有标注框的位置造静止帧）``视频统计非静止帧和静止帧的结果


#### 1.1.1. 20230112实验结果分析

``在不同split来验证是否有效``

根据上周的实验结果，如下图所示。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="file\experiment_20230112.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">实验结果</div>
</center>


以0.4概率移除静止帧在大多数指标更好。认为是因为静止帧在视频中占据比较长的位置，所以相当于增加了学习的样本。同时，过多引入静止帧会造成下降，因为过拟合了简单样本。

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

[GitHub Repo](https://github.com/ljwztc/CLIP-Driven-Universal-Model)

[Paper-Jan 2](https://arxiv.org/abs/2301.00785)

``简介`` 在多个部位的癌症数据集上学习；CLIP。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="file\clip_driven_universal_model.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">CILP-Driven Universal Model</div>
</center>


### 2.2. MICCAI22 Part III Colonoscopy部分

[MICCAI 2022 Part III Colonoscopy](https://github.com/xjtulyc/MICCAI2022_paper_reading/blob/main/Part%20III/notes.md)

## 3. 人工智能芯片设计导论大作业

[作业要求](https://github.com/xjtulyc/weekly-summary-2023/blob/main/20230119/file/1\)%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E8%8A%AF%E7%89%87%E8%AE%BE%E8%AE%A1%E5%AF%BC%E8%AE%BA%E8%AF%BE%E7%A8%8B%E4%BD%9C%E4%B8%9A.pdf)：
- 课程大作业截止日期3月31日（我测，怎么这么难）
