# 周报-20230314

1. 比较mem和无mem的推理结果
2. 比较静止帧甲乳合并的训练结果

**静止帧甲乳合并训练结果** （无mem）

TUS rate = 0.0, BUS rate = 0.0

|Type|Split|AP50|P@0.7|FP|R@16|#Video|
|---|---                      |  --- | --- |   ---   |  --- | --- |
|Dynamic| breast_ALL@20221108-145033 | 86.71 | 92.01 | 0.7632 | 96.82 |  1440  |
|Static| breast_ALL@20221108-145033 | 94.06 | 98.32 | 0.1982 | 96.79 |  1440  |

TUS rate = 0.1, BUS rate = 0.4

|Type|Split|AP50|P@0.7|FP|R@16|#Video|
|---|---                      |  --- | --- |   ---   |  --- | --- |
|Dynamic| breast_ALL@20221108-145033 | 86.39 | 91.84 | 0.7660 | 97.04 |  1440  |
|Static| breast_ALL@20221108-145033 | 94.97 | 98.77 | 0.1164 | 97.04 |  1440  |

**有mem和无mem的推理结果**

正在推理中

有mem

TUS rate = 0.0, BUS rate = 0.0 49018

TUS rate = 0.1, BUS rate = 0.4 49019

无mem

TUS rate = 0.0, BUS rate = 0.0 49017

TUS rate = 0.1, BUS rate = 0.4 49038

目前速度

```
30W_t_0_b_0_20w
967
30W_t_0_b_0_static_mem_infer_20w
1088
30W_t_1_b_4_20w
1391
30W_t_1_b_4_static_mem_infer_20w
1020
BUS 30W_t_0_b_0
1440
TUS 30W_t_0_b_0
909
BUS 30W_t_1_b_4 
1440
```