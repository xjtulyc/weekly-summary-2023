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
- 课程大作业截止日期3月31日

进度控制：
- 复习课程PPT 2023-01-14 to 2023-01-31
- 作业I 2023-02-01 to 2023-02-28
- 作业II 2023-03-01 to 2023-03-31

### 3.1. PPT
<p style="text-indent:24.1pt"><span style="font-size:14px">秋季课程《人工智能芯片设计导论》（2021- ）</span></p>

<p style="text-indent:24.1pt"><span style="font-size:14px"><span style="font-size: 14px;">指导教师：任鹏举、赵文哲、夏天</span></span></p>

<p style="text-indent:24.1pt"><a href="/documents/1796039/0/01-intro-2022-AI+Chip.pdf/244e0f1c-08b0-42e2-a75f-715f0bca113d?t=1665317459616"><span style="font-size: 14px;"><strong>01-Intro </strong></span></a><span style="font-size: 14px;"><strong>【</strong></span><span style="font-size: 14px;"><strong>Updated 2022.10.09】</strong></span></p>

<p style="text-indent:24.1pt"><a href="/documents/1796039/0/02-Data+Stream+App+and+Various+Architectures-2022+AI+Chip.pdf/f511d58e-5bce-3f56-941c-6161a82c8fec?t=1665317535221"><span style="font-size: 14px;"><strong>02-Data Stream App and Various Architectures</strong></span></a>&nbsp;<span style="font-size: 14px;"><strong>【</strong></span><span style="font-size: 14px;"><strong>Updated 2022.10.09】</strong></span></p>

<p style="text-indent:24.1pt"><a href="/documents/1796039/0/03-Graphical+Representations-2022+AI+Chip.pdf/bc33a5d8-e8ec-6945-173f-a9de42651293?t=1665317567532"><span style="font-size: 14px;"><strong>03-Graphical Representations</strong></span></a>&nbsp;<span style="font-size: 14px;"><span style="font-size: 14px;"><strong>【</strong></span><strong>Updated 2022.10.09】</strong></span></p>

<p style="text-indent:24.1pt"><a href="/documents/1796039/0/04-Iteration+Bound-2022+AI+Chip.pdf/db87a0cd-382c-9b93-951e-99dd83200ed8?t=1665317583069"><span style="font-size: 14px;"><strong>04-Iteration Bound</strong></span></a>&nbsp;<span style="font-size: 14px;"><span style="font-size: 14px;"><strong>【</strong></span><strong>Updated 2022.10.09】</strong></span></p>

<p style="text-indent:24.1pt"><a href="/documents/1796039/0/05-Retiming+and+Pipelining-2022+AI+Chip.pdf/6c4196dd-b6c5-52f0-787a-f0784b066b9f?t=1665317601086"><span style="font-size: 14px;"><strong>05-Retiming and Pipelining</strong></span></a>&nbsp;<span style="font-size: 14px;"><strong>【</strong></span><span style="font-size: 14px;"><strong>Updated 2022.10.09】</strong></span></p>

<p style="text-indent:24.1pt"><span style="font-size: 14px;"><strong><a href="/documents/1796039/0/06-Parallel+Architecture+%28Unfolding%29-2022+AI+Chip.pdf/2400d1e5-26ea-b008-29ac-029a803e7ff8?t=1667513426580">06-Parallel Architecture (Unfolding) </a></strong><span style="font-size: 14px;"><strong>【</strong></span><strong>Updated 2022.11.04</strong><span style="font-size: 14px;"><strong>】</strong></span></span></p>

<p style="text-indent:24.1pt"><span style="font-size: 14px;"><strong><a href="/documents/1796039/0/07-Resource+Sharing+%28Folding%29-2022+AI+Chip.pdf/afd607cd-29cc-c2a1-5f39-8266531e28bf?t=1667513450208">07-Resource Sharing (Folding)</a> </strong><span style="font-size: 14px;"><strong>【</strong></span><strong>Updated 2022.11.04</strong><span style="font-size: 14px;"><strong>】</strong></span></span></p>

<p style="text-indent:24.1pt"><span style="font-size: 14px;"><strong><a href="/documents/1796039/0/08-Scheduling+and+Resource+Allocation+-2022+AI+Chip.pdf/42bb72f1-fc89-54ac-c666-7fb92a72fe3c?t=1667513468344">08-Scheduling and Resource Allocation</a> </strong><span style="font-size: 14px;"><strong>【</strong></span><strong>Updated 2022.11.04</strong><span style="font-size: 14px;"><strong>】</strong></span></span></p>

<p style="text-indent:24.1pt"><span style="font-size: 14px;"><strong><a href="/documents/1796039/0/09-Systolic+Array-2022+AI+Chip.pdf/43cf86d0-367d-c3ec-6fe4-e3651c095291?t=1667513482765">09-Systolic Array</a> </strong><span style="font-size: 14px;"><strong>【</strong></span><strong>Updated 2022.11.04</strong><span style="font-size: 14px;"><strong>】</strong></span></span></p>

<p style="text-indent:24.1pt">&nbsp;</p>