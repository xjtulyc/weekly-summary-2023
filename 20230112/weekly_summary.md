# 周报-20230112

## 1. 项目进展Ultrasound_vid
### 1.1. 使用基础配置跑baseline
```
--config-file configs/RDN-LSTM/BUS_BasicConfig_StaticFrame.yaml
```
修改``config.py``
```
# static frame
cfg.STATIC_FRAME = CN() # 创建树的节点
cfg.STATIC_FRAME.RATE = 0.0
```

修改``frame_sampler.py``的``line 148``和``line 180``
```
self.static_frame_rate = None
if cfg.STATIC_FRAME.RATE is not None:
    self.static_frame_rate = cfg.STATIC_FRAME.RATE
else:
    self.static_frame_rate = 0.0

```
```
# remove static frames
# 静止帧
rate = self.static_frame_rate
sample_index_idx_base_rmstatic = list(
    filter(
        lambda i: i not in video_anno_dict["video_info"]["static_frames"],
        sample_frame_idx_base,
    )
)
if len(sample_index_idx_base_rmstatic) > 30 and np.random.rand() > rate:  # 以rate的概率去除静止帧; torch
    sample_frame_idx_base = sample_index_idx_base_rmstatic
```

### 1.2. 以一定概率放入静止帧训练

· 以[**0.0**, **0.3**, **0.4**, **0.5**, **0.6**, **0.7**]的概率随机选择静止帧训练模型，并比较性能

joblog 36405 rate 0.0
```
[01/08 18:24:58 vid.evaluation.video_evaluation]: Evaluating mmdet-style AP on 'breast_ALL@20221108-145033' dataset
[01/08 18:24:58 vid.evaluation.video_evaluation]: The fixed_recall is [0.7]
[01/08 18:36:29 vid.evaluation.video_evaluation]: Calculating FP rate on 'breast_ALL@20221108-145033' dataset
[01/08 18:36:29 vid.evaluation.video_evaluation]: Prob threshold is [0.8555757462978364]
[01/08 18:46:33 vid.evaluation.video_evaluation]: Evaluating mmdet-style AP on 'breast_ALL@20221108-145033' dataset
[01/08 19:03:15 vid.evaluation.video_evaluation]: >>> Dataset dicts temp file removed!
[01/08 19:03:40 ultrasound_vid]: Evaluation results for breast_ALL@20221108-145033 in csv format:
[01/08 19:03:40 d2.evaluation.testing]: copypaste: Task: Recall
[01/08 19:03:40 d2.evaluation.testing]: copypaste: R@16
[01/08 19:03:40 d2.evaluation.testing]: copypaste: 0.9723
[01/08 19:03:40 d2.evaluation.testing]: copypaste: Task: Precision
[01/08 19:03:40 d2.evaluation.testing]: copypaste: P@R0.7,multi_P@R0.7
[01/08 19:03:40 d2.evaluation.testing]: copypaste: 0.9159,0.8452
[01/08 19:03:40 d2.evaluation.testing]: copypaste: Task: Scale Precision
[01/08 19:03:40 d2.evaluation.testing]: copypaste: P@R0.7_0,P@R0.7_1,P@R0.7_2,P@R0.7_3,P@R0.7_4
[01/08 19:03:40 d2.evaluation.testing]: copypaste: 0.6447,0.8457,0.8950,0.9327,0.9541
[01/08 19:03:40 d2.evaluation.testing]: copypaste: Task: Average Precision
[01/08 19:03:40 d2.evaluation.testing]: copypaste: AP50,multi_AP50
[01/08 19:03:40 d2.evaluation.testing]: copypaste: 0.8673,0.8106
[01/08 19:03:40 d2.evaluation.testing]: copypaste: Task: Scale Average Precision
[01/08 19:03:40 d2.evaluation.testing]: copypaste: AP50_0,AP50_1,AP50_2,AP50_3,AP50_4
[01/08 19:03:40 d2.evaluation.testing]: copypaste: 0.7204,0.8174,0.8675,0.8893,0.8998
[01/08 19:03:40 d2.evaluation.testing]: copypaste: Task: FP stat
[01/08 19:03:40 d2.evaluation.testing]: copypaste: FP/min
[01/08 19:03:40 d2.evaluation.testing]: copypaste: 0.8642
```

joblog 36418 rate 0.3

```
[01/11 01:09:16 vid.evaluation.video_evaluation]: Evaluating mmdet-style AP on 'breast_ALL@20221108-145033' dataset
[01/11 01:09:16 vid.evaluation.video_evaluation]: The fixed_recall is [0.7]
[01/11 01:24:24 vid.evaluation.video_evaluation]: Calculating FP rate on 'breast_ALL@20221108-145033' dataset
[01/11 01:24:24 vid.evaluation.video_evaluation]: Prob threshold is [0.8638185739517212]
[01/11 01:33:50 vid.evaluation.video_evaluation]: Evaluating mmdet-style AP on 'breast_ALL@20221108-145033' dataset
[01/11 01:50:06 vid.evaluation.video_evaluation]: >>> Dataset dicts temp file removed!
[01/11 01:50:33 ultrasound_vid]: Evaluation results for breast_ALL@20221108-145033 in csv format:
[01/11 01:50:33 d2.evaluation.testing]: copypaste: Task: Recall
[01/11 01:50:33 d2.evaluation.testing]: copypaste: R@16
[01/11 01:50:33 d2.evaluation.testing]: copypaste: 0.9733
[01/11 01:50:33 d2.evaluation.testing]: copypaste: Task: Precision
[01/11 01:50:33 d2.evaluation.testing]: copypaste: P@R0.7,multi_P@R0.7
[01/11 01:50:33 d2.evaluation.testing]: copypaste: 0.9193,0.8380
[01/11 01:50:33 d2.evaluation.testing]: copypaste: Task: Scale Precision
[01/11 01:50:33 d2.evaluation.testing]: copypaste: P@R0.7_0,P@R0.7_1,P@R0.7_2,P@R0.7_3,P@R0.7_4
[01/11 01:50:33 d2.evaluation.testing]: copypaste: 0.6437,0.8456,0.9054,0.9340,0.9561
[01/11 01:50:33 d2.evaluation.testing]: copypaste: Task: Average Precision
[01/11 01:50:33 d2.evaluation.testing]: copypaste: AP50,multi_AP50
[01/11 01:50:33 d2.evaluation.testing]: copypaste: 0.8682,0.8090
[01/11 01:50:33 d2.evaluation.testing]: copypaste: Task: Scale Average Precision
[01/11 01:50:33 d2.evaluation.testing]: copypaste: AP50_0,AP50_1,AP50_2,AP50_3,AP50_4
[01/11 01:50:33 d2.evaluation.testing]: copypaste: 0.7212,0.8162,0.8683,0.8861,0.9049
[01/11 01:50:33 d2.evaluation.testing]: copypaste: Task: FP stat
[01/11 01:50:33 d2.evaluation.testing]: copypaste: FP/min
[01/11 01:50:33 d2.evaluation.testing]: copypaste: 0.7705
```

joblog 36429 rate 0.4

```
[01/11 05:50:17 vid.evaluation.video_evaluation]: Evaluating mmdet-style AP on 'breast_ALL@20221108-145033' dataset
[01/11 05:50:17 vid.evaluation.video_evaluation]: The fixed_recall is [0.7]
[01/11 06:02:08 vid.evaluation.video_evaluation]: Calculating FP rate on 'breast_ALL@20221108-145033' dataset
[01/11 06:02:08 vid.evaluation.video_evaluation]: Prob threshold is [0.8671300768852235]
[01/11 06:11:32 vid.evaluation.video_evaluation]: Evaluating mmdet-style AP on 'breast_ALL@20221108-145033' dataset
[01/11 06:26:28 vid.evaluation.video_evaluation]: >>> Dataset dicts temp file removed!
[01/11 06:26:54 ultrasound_vid]: Evaluation results for breast_ALL@20221108-145033 in csv format:
[01/11 06:26:54 d2.evaluation.testing]: copypaste: Task: Recall
[01/11 06:26:54 d2.evaluation.testing]: copypaste: R@16
[01/11 06:26:54 d2.evaluation.testing]: copypaste: 0.9730
[01/11 06:26:54 d2.evaluation.testing]: copypaste: Task: Precision
[01/11 06:26:54 d2.evaluation.testing]: copypaste: P@R0.7,multi_P@R0.7
[01/11 06:26:54 d2.evaluation.testing]: copypaste: 0.9207,0.8486
[01/11 06:26:54 d2.evaluation.testing]: copypaste: Task: Scale Precision
[01/11 06:26:54 d2.evaluation.testing]: copypaste: P@R0.7_0,P@R0.7_1,P@R0.7_2,P@R0.7_3,P@R0.7_4
[01/11 06:26:54 d2.evaluation.testing]: copypaste: 0.6542,0.8419,0.9057,0.9390,0.9568
[01/11 06:26:54 d2.evaluation.testing]: copypaste: Task: Average Precision
[01/11 06:26:54 d2.evaluation.testing]: copypaste: AP50,multi_AP50
[01/11 06:26:54 d2.evaluation.testing]: copypaste: 0.8700,0.8111
[01/11 06:26:54 d2.evaluation.testing]: copypaste: Task: Scale Average Precision
[01/11 06:26:54 d2.evaluation.testing]: copypaste: AP50_0,AP50_1,AP50_2,AP50_3,AP50_4
[01/11 06:26:54 d2.evaluation.testing]: copypaste: 0.7222,0.8184,0.8705,0.8896,0.9058
[01/11 06:26:54 d2.evaluation.testing]: copypaste: Task: FP stat
[01/11 06:26:54 d2.evaluation.testing]: copypaste: FP/min
[01/11 06:26:54 d2.evaluation.testing]: copypaste: 0.7374
```

joblog 36407 rate 0.5
```
[01/08 21:52:53 vid.evaluation.video_evaluation]: Evaluating mmdet-style AP on 'breast_ALL@20221108-145033' dataset
[01/08 21:52:53 vid.evaluation.video_evaluation]: The fixed_recall is [0.7]
[01/08 22:05:50 vid.evaluation.video_evaluation]: Calculating FP rate on 'breast_ALL@20221108-145033' dataset
[01/08 22:05:50 vid.evaluation.video_evaluation]: Prob threshold is [0.8629564642906189]
[01/08 22:15:44 vid.evaluation.video_evaluation]: Evaluating mmdet-style AP on 'breast_ALL@20221108-145033' dataset
[01/08 22:32:06 vid.evaluation.video_evaluation]: >>> Dataset dicts temp file removed!
[01/08 22:32:36 ultrasound_vid]: Evaluation results for breast_ALL@20221108-145033 in csv format:
[01/08 22:32:36 d2.evaluation.testing]: copypaste: Task: Recall
[01/08 22:32:36 d2.evaluation.testing]: copypaste: R@16
[01/08 22:32:36 d2.evaluation.testing]: copypaste: 0.9747
[01/08 22:32:36 d2.evaluation.testing]: copypaste: Task: Precision
[01/08 22:32:36 d2.evaluation.testing]: copypaste: P@R0.7,multi_P@R0.7
[01/08 22:32:36 d2.evaluation.testing]: copypaste: 0.9177,0.8383
[01/08 22:32:36 d2.evaluation.testing]: copypaste: Task: Scale Precision
[01/08 22:32:36 d2.evaluation.testing]: copypaste: P@R0.7_0,P@R0.7_1,P@R0.7_2,P@R0.7_3,P@R0.7_4
[01/08 22:32:36 d2.evaluation.testing]: copypaste: 0.6513,0.8383,0.9046,0.9328,0.9537
[01/08 22:32:36 d2.evaluation.testing]: copypaste: Task: Average Precision
[01/08 22:32:36 d2.evaluation.testing]: copypaste: AP50,multi_AP50
[01/08 22:32:36 d2.evaluation.testing]: copypaste: 0.8693,0.8018
[01/08 22:32:36 d2.evaluation.testing]: copypaste: Task: Scale Average Precision
[01/08 22:32:36 d2.evaluation.testing]: copypaste: AP50_0,AP50_1,AP50_2,AP50_3,AP50_4
[01/08 22:32:36 d2.evaluation.testing]: copypaste: 0.7171,0.8156,0.8702,0.8869,0.9061
[01/08 22:32:36 d2.evaluation.testing]: copypaste: Task: FP stat
[01/08 22:32:36 d2.evaluation.testing]: copypaste: FP/min
[01/08 22:32:36 d2.evaluation.testing]: copypaste: 0.7906
```

joblog 36466 rate 0.6
```
[01/13 08:32:48 vid.evaluation.video_evaluation]: Evaluating mmdet-style AP on 'breast_ALL@20221108-145033' dataset
[01/13 08:32:48 vid.evaluation.video_evaluation]: The fixed_recall is [0.7]
[01/13 08:44:41 vid.evaluation.video_evaluation]: Calculating FP rate on 'breast_ALL@20221108-145033' dataset
[01/13 08:44:41 vid.evaluation.video_evaluation]: Prob threshold is [0.8508420825004579]
[01/13 08:54:31 vid.evaluation.video_evaluation]: Evaluating mmdet-style AP on 'breast_ALL@20221108-145033' dataset
[01/13 09:10:25 vid.evaluation.video_evaluation]: >>> Dataset dicts temp file removed!
[01/13 09:10:51 ultrasound_vid]: Evaluation results for breast_ALL@20221108-145033 in csv format:
[01/13 09:10:51 d2.evaluation.testing]: copypaste: Task: Recall
[01/13 09:10:51 d2.evaluation.testing]: copypaste: R@16
[01/13 09:10:51 d2.evaluation.testing]: copypaste: 0.9727
[01/13 09:10:51 d2.evaluation.testing]: copypaste: Task: Precision
[01/13 09:10:51 d2.evaluation.testing]: copypaste: P@R0.7,multi_P@R0.7
[01/13 09:10:51 d2.evaluation.testing]: copypaste: 0.9163,0.8285
[01/13 09:10:51 d2.evaluation.testing]: copypaste: Task: Scale Precision
[01/13 09:10:51 d2.evaluation.testing]: copypaste: P@R0.7_0,P@R0.7_1,P@R0.7_2,P@R0.7_3,P@R0.7_4
[01/13 09:10:51 d2.evaluation.testing]: copypaste: 0.6288,0.8383,0.9005,0.9337,0.9545
[01/13 09:10:51 d2.evaluation.testing]: copypaste: Task: Average Precision
[01/13 09:10:51 d2.evaluation.testing]: copypaste: AP50,multi_AP50
[01/13 09:10:51 d2.evaluation.testing]: copypaste: 0.8676,0.8004
[01/13 09:10:51 d2.evaluation.testing]: copypaste: Task: Scale Average Precision
[01/13 09:10:51 d2.evaluation.testing]: copypaste: AP50_0,AP50_1,AP50_2,AP50_3,AP50_4
[01/13 09:10:51 d2.evaluation.testing]: copypaste: 0.7125,0.8141,0.8678,0.8901,0.9034
[01/13 09:10:51 d2.evaluation.testing]: copypaste: Task: FP stat
[01/13 09:10:51 d2.evaluation.testing]: copypaste: FP/min
[01/13 09:10:51 d2.evaluation.testing]: copypaste: 0.8390
```

joblog 36467 rate 0.7
```
[01/13 13:24:10 vid.evaluation.video_evaluation]: Evaluating mmdet-style AP on 'breast_ALL@20221108-145033' dataset
[01/13 13:24:10 vid.evaluation.video_evaluation]: The fixed_recall is [0.7]
[01/13 13:35:59 vid.evaluation.video_evaluation]: Calculating FP rate on 'breast_ALL@20221108-145033' dataset
[01/13 13:35:59 vid.evaluation.video_evaluation]: Prob threshold is [0.8595068097114563]
[01/13 13:46:07 vid.evaluation.video_evaluation]: Evaluating mmdet-style AP on 'breast_ALL@20221108-145033' dataset
[01/13 14:02:28 vid.evaluation.video_evaluation]: >>> Dataset dicts temp file removed!
[01/13 14:02:51 ultrasound_vid]: Evaluation results for breast_ALL@20221108-145033 in csv format:
[01/13 14:02:51 d2.evaluation.testing]: copypaste: Task: Recall
[01/13 14:02:51 d2.evaluation.testing]: copypaste: R@16
[01/13 14:02:51 d2.evaluation.testing]: copypaste: 0.9735
[01/13 14:02:51 d2.evaluation.testing]: copypaste: Task: Precision
[01/13 14:02:51 d2.evaluation.testing]: copypaste: P@R0.7,multi_P@R0.7
[01/13 14:02:51 d2.evaluation.testing]: copypaste: 0.9172,0.8320
[01/13 14:02:51 d2.evaluation.testing]: copypaste: Task: Scale Precision
[01/13 14:02:51 d2.evaluation.testing]: copypaste: P@R0.7_0,P@R0.7_1,P@R0.7_2,P@R0.7_3,P@R0.7_4
[01/13 14:02:51 d2.evaluation.testing]: copypaste: 0.6327,0.8366,0.9042,0.9346,0.9557
[01/13 14:02:51 d2.evaluation.testing]: copypaste: Task: Average Precision
[01/13 14:02:51 d2.evaluation.testing]: copypaste: AP50,multi_AP50
[01/13 14:02:51 d2.evaluation.testing]: copypaste: 0.8685,0.8034
[01/13 14:02:51 d2.evaluation.testing]: copypaste: Task: Scale Average Precision
[01/13 14:02:51 d2.evaluation.testing]: copypaste: AP50_0,AP50_1,AP50_2,AP50_3,AP50_4
[01/13 14:02:51 d2.evaluation.testing]: copypaste: 0.7135,0.8113,0.8708,0.8895,0.9054
[01/13 14:02:51 d2.evaluation.testing]: copypaste: Task: FP stat
[01/13 14:02:51 d2.evaluation.testing]: copypaste: FP/min
[01/13 14:02:51 d2.evaluation.testing]: copypaste: 0.8000
```

### 1.3. 衡量静止帧训练程度的评价指标

对静止帧统计AP

## 2. 论文阅读
### 2.1. 乳腺影像报告与数据系统图谱


## 3. 毕业设计

### 3.1. 开题答辩
开学一两周

### 3.2. 开题报告
#### 3.2.1. 对指导教师下达的课题任务的学习与理解
#### 3.2.2. 阅读文献资料进行调研的综述（10篇左右）
#### 3.2.3. 根据任务书的任务及文献调研结果，初步拟定的执行（实施）方案（含具体进度计划）
[开题报告初稿（未完成）](https://github.com/xjtulyc/weekly-summary-2023/blob/main/20230112/file/%E5%BC%80%E9%A2%98%E6%8A%A5%E5%91%8A.pdf)

### 3.3. 开题PPT
1. 搞懂代码
2. 学会写配置文件
3. 视频目标检测的评价指标
4. 问题背景、文献综述 2 min
5. 数据集统计，可能或已经出现的问题分析 2min
6. 工作安排 1 min

## 4. 论文汇报
### 4.1. MedKLIP: Medical Knowledge Enhanced Language-Image Pre-Training

总结：idea不错，用结构化文本降低学习病历文本的难度；但是实验做的不太好。

### 4.2. 整理MICCIA22任务和数据集，看看值不值得做，是否开源；纯分类的任务可以放一下；**找好的Task**

[MACCIA2022 Paper Reading](https://github.com/xjtulyc/MACCIA2022_paper_reading)