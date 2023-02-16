# 周报-20230216
为了提升检测模型在静止帧上的表现，提出可以在训练时保留一部分静止帧的猜想，即验证调整移除静止帧概率是否可以提升模型静止帧表现。

为了验证移除静止帧概率对模型FP（假阳性）的影响，首先在乳腺数据集上以0.0，0.3, 0.4, 0.5,0.6, 0.7的移除静止帧概率训练模型，并在静止帧、运动帧和全部帧上比较，发现以0.4概率移除静止帧不光能够提升静止帧上的表现，也能提高运动帧上的FP。可以佐证调整移除静止帧概率可以在略微损失运动帧精度的情况下，提升静止帧表现的猜想。

在甲状腺数据集上以0.4的概率移除静止帧训练，发现虽然在静止帧上表现上升，但运动帧的表现下降/(latest)（$FP\uparrow:1.3 \rightarrow  2.0/min$），并不能完全复现乳腺数据集上的结果。虽然目前在甲状腺数据集上并不充分，但是可以说明移除静止帧的最优概率和数据集分布有关，简单调整移除概率并不能在甲-乳数据集上解决静止帧的问题。

为了明确移除部分静止帧的策略在甲状腺数据集上的影响，下周将在甲状腺数据集上训练0.2, 0.4, 0.6, 0.8, 1.0移除静止帧的检测模型的表现。

## 1. Ultrasound项目
### 1.1. 实验数据整理

**分医院/最好的比例/全部医院合并结果**
**乳腺+FPN/MASTER，观察结论是否成立**

**先验证甲状腺（FPN）/公平对比**
**Breast Ultrasound**
不加入静止帧实验结果
| Split                 |  AP50 | P@0.7 |   FP   |  R@16 | #Video |
|:---|---|---|---|---|---:|
| 北京大学肿瘤医院_阳性 | **86.91** | **92.94**| **0.5265** | 94.42 |   21   |
| 郑州大学第一附属医院_阳性 | 81.65 | 84.70 | 1.5282 | 94.34 |   19   |
| 宝清县人民医院_阳性 | 84.78 | 85.67 | 0.5572 | 96.84 |   22   |
| 云南省大理大学第一附属医院_阳性 | 50.46 |  9.18 | 152.0693 | 82.81 |   5    |
| 大连医科大学附属第一医院_阳性 | **89.80** | **96.18** | **0.3011** | 96.54 |   36   |
| 广东省中医院_阳性 | **90.23** | **94.22** | 1.0677 | 96.50 |   76   |
| 中国科学院大学附属肿瘤医院_阳性 | **87.42** | **92.68** | **1.9898** | **95.91** |   73   |
| 四川省广元市中心医院_阳性 | 91.78 | 92.96 | **0.6171** | **99.19** |   8    |
| 南昌市第三医院_阴性 | 0.00 |  0.00 | 1.2895 | 0.00 |  155   |
| 南昌市第三医院_阳性 | **87.67** | **93.19** | 0.5539 | 96.33 |  476   |
| 铁岭市中心医院_阳性 | 95.71 | **98.39** | **0.4508** | 99.80 |   7    |
| 云南省肿瘤医院_阳性 | 92.84 | **97.99** | **0.1031** | **98.83** |   25   |

以0.4移除静止帧结果（0202实验）
| Split                 |  AP50 | P@0.7 |   FP   |  R@16 | #Video |
|:---|---|---|---|---|---:|
| 北京大学肿瘤医院_阳性 | 86.07 | 90.33 | 0.6911 | **97.95** |   21   |
| 郑州大学第一附属医院_阳性 | **84.68** | **91.09** | **1.2662** | **96.55** |   19   |
| 宝清县人民医院_阳性 | **89.15** | **91.95** | **0.4986** | **98.78** |   22   |
| 云南省大理大学第一附属医院_阳性 | **54.24** | **23.70** | **42.2757** | **85.67** |   5    |
| 大连医科大学附属第一医院_阳性 | 89.57 | 95.43 | 0.4517 | **96.91** |   36   |
| 广东省中医院_阳性 | 89.90 | 93.88 | **0.9798** | **98.19** |   76   |
| 中国科学院大学附属肿瘤医院_阳性 | 86.40 | 92.04 | 2.2310 | 95.59 |   73   |
| 四川省广元市中心医院_阳性 | **92.35** | **95.97** | 0.7714 | 98.85 |   8    |
| 南昌市第三医院_阴性 | 0.00 |  0.00 | **0.5826** | 0.00 |  155   |
| 南昌市第三医院_阳性 | 87.62 | 92.85 | **0.5519** | **96.67** |  476   |
| 铁岭市中心医院_阳性 | **95.82** | 97.64 | 0.4508 | **99.87** |   7    |
| 云南省肿瘤医院_阳性 | **93.26** | 97.66 | 0.1351 | 98.58 |   25   |
### 1.2. 在TUS数据集上搜索参数的影响

```shell
[02/15 19:05:18 vid.evaluation.video_evaluation]: Evaluating mmdet-style AP on 'thyroid_ALL@20221104-110210' dataset
[02/15 19:05:18 vid.evaluation.video_evaluation]: The fixed_recall is [0.7]
[02/15 19:09:01 vid.evaluation.video_evaluation]: Calculating FP rate on 'thyroid_ALL@20221104-110210' dataset
[02/15 19:09:01 vid.evaluation.video_evaluation]: Prob threshold is [0.9150264382362366]
[02/15 19:11:16 vid.evaluation.video_evaluation]: Evaluating mmdet-style AP on 'thyroid_ALL@20221104-110210' dataset
[02/15 19:16:23 vid.evaluation.video_evaluation]: >>> Dataset dicts temp file removed!
[02/15 19:16:31 ultrasound_vid]: Evaluation results for thyroid_ALL@20221104-110210 in csv format:
[02/15 19:16:31 d2.evaluation.testing]: copypaste: Task: Recall
[02/15 19:16:31 d2.evaluation.testing]: copypaste: R@16
[02/15 19:16:31 d2.evaluation.testing]: copypaste: 0.9535
[02/15 19:16:31 d2.evaluation.testing]: copypaste: Task: Precision
[02/15 19:16:31 d2.evaluation.testing]: copypaste: P@R0.7,multi_P@R0.7
[02/15 19:16:31 d2.evaluation.testing]: copypaste: 0.9194,0.9051
[02/15 19:16:31 d2.evaluation.testing]: copypaste: Task: Scale Precision
[02/15 19:16:31 d2.evaluation.testing]: copypaste: P@R0.7_0,P@R0.7_1,P@R0.7_2,P@R0.7_3,P@R0.7_4
[02/15 19:16:31 d2.evaluation.testing]: copypaste: 0.7282,0.9337,0.9781,0.9739,0.9832
[02/15 19:16:31 d2.evaluation.testing]: copypaste: Task: Average Precision
[02/15 19:16:31 d2.evaluation.testing]: copypaste: AP50,multi_AP50
[02/15 19:16:31 d2.evaluation.testing]: copypaste: 0.8635,0.8425
[02/15 19:16:31 d2.evaluation.testing]: copypaste: Task: Scale Average Precision
[02/15 19:16:31 d2.evaluation.testing]: copypaste: AP50_0,AP50_1,AP50_2,AP50_3,AP50_4
[02/15 19:16:31 d2.evaluation.testing]: copypaste: 0.7383,0.8771,0.9262,0.9484,0.9569
[02/15 19:16:31 d2.evaluation.testing]: copypaste: Task: FP stat
[02/15 19:16:31 d2.evaluation.testing]: copypaste: FP/min
[02/15 19:16:31 d2.evaluation.testing]: copypaste: 2.0462
```

### 1.3. 在静止帧上检查

调用逻辑

``tools/eval_pred.py`` $\rightarrow$ ``main()``（三种模式分别执行一次） $\rightarrow$ ``eval_pred()`` （传入形参选择模式） $\rightarrow$ ``UltrasoundVideoDetectionEvaluator.evaluate``（传入形参选择模式）

在``ultrasound_vid/evaluation/video_evaluation.py``中修改``calculate_mmdet_ap``，支持三种模式：

1. 只在静止帧上验证；
2. 只在运动帧上验证；
3. 在所有帧上验证。

**正在等待结果**

<details>
<summary>tools/eval_pred.py</summary>

```python
import os
from glob import glob

import pandas as pd
import prettytable as pt
import torch
from ultrasound_vid.evaluation.evaluator import UltrasoundVideoDetectionEvaluator


# setup_logger(name="ultrasound_vid")
# setup_logger(name="detectron2")

def eval_pred(dataset_name,
              predictions,
              remove_static = True,
              static_only = False,
              device=None,
              hospital=None,
              negative=None,
              negative_fp_thresh=None):
    evaluator = UltrasoundVideoDetectionEvaluator(dataset_name=[dataset_name])
    evaluator.predictions = predictions
    results = evaluator.evaluate(fixed_thresh=negative_fp_thresh,
                                 remove_static = remove_static,
                                 static_only = static_only)
    split = dataset_name
    if device is not None:
        split = device
    elif hospital is not None:
        split = hospital
        if negative is not None:
            split_negative = "阴性" if negative else "阳性"
            split += ("_" + split_negative)
    elif negative is not None:
        split = "阴性" if negative else "阳性"
    # print("*"*10, device, len(predictions), "*"*10)
    tb = pt.PrettyTable()
    tb.field_names = ["Split", "AP50", "P@0.7", "FP", "R@16", "#Video"]
    tb.align["Split"] = "l"
    AP = results["Average Precision"]["AP50"]
    prec = results["Precision"]["P@R0.7"]
    rec = results["Recall"]["R@16"]
    fp = results["FP stat"]["FP/min"]
    tb.add_row([split, f"{AP * 100:.2f}", f"{prec * 100:.2f}", f"{fp:.4f}", f"{rec * 100:.2f}", len(predictions)])
    print(tb)


def main(file_path,
         dataset_name,
         csv_file="",
         by_device=False,
         by_hospital=False,
         by_negative=False,
         negative_fp_thresh=None,
         remove_static = True,
         static_only = False,):
    """根据已有的inference结果进行evaluation
    Params:
        file_path   : str, 存放inference结果的目录
        dataset_name: str, 数据集名称
        csv_file    : str, csv文件路径
        by_device   : bool, 是否根据机型统计evaluation结果
        by_hospital : bool, 是否根据医院统计evaluation结果
        by_negative : bool, 是否根据阴性/阳性视频统计evaluation结果
        negative_fp_thresh : float, 仅在统计阴性视频的FP时用到, 作为判断FP的阈值
    """
    prediction_files = glob(os.path.join(file_path, "*.pth"))
    print(f"loaded {len(prediction_files)} pred videos")
    predictions = {}
    for f in prediction_files:
        predictions.update(torch.load(f, map_location="cpu"))
    if by_device:
        dataset_df = pd.read_csv(csv_file)
        devices = dataset_df.device.unique()
        for device in devices:
            pred_device = {k: v for k, v in predictions.items() if device in k}
            if len(pred_device) == 0:
                print("*" * 10, device, len(pred_device), "*" * 10)
                continue
            eval_pred(dataset_name,
                      pred_device,
                      device=device,
                      remove_static = remove_static,
                      static_only = static_only)
    elif by_hospital:
        assert not by_device
        dataset_df = pd.read_csv(csv_file)
        hospitals = dataset_df.hospital.unique()
        for hospital in hospitals:
            if by_negative:
                assert negative_fp_thresh is not None
                df_hospital = dataset_df[dataset_df.hospital == hospital]
                for negative in [True, False]:
                    hospital_keys = list(df_hospital[df_hospital.neg == negative].db_key)
                    pred_hospital = {k: v for k, v in predictions.items() if k in hospital_keys}
                    if len(pred_hospital) == 0: continue
                    eval_pred(dataset_name,
                              pred_hospital,
                              hospital=hospital,
                              negative=negative,
                              negative_fp_thresh=negative_fp_thresh if negative else None,
                              remove_static=remove_static,
                              static_only=static_only
                              )
            else:
                hospital_keys = list(dataset_df[dataset_df.hospital == hospital].db_key)
                pred_hospital = {k: v for k, v in predictions.items() if k in hospital_keys}
                if len(pred_hospital) == 0: continue
                eval_pred(dataset_name,
                          pred_hospital,
                          hospital=hospital,
                          remove_static = remove_static,
                          static_only = static_only)
    elif by_negative:
        dataset_df = pd.read_csv(csv_file)
        assert negative_fp_thresh is not None
        for negative in [True, False]:
            negative_keys = list(dataset_df[dataset_df.neg == negative].db_key)
            pred_negative = {k: v for k, v in predictions.items() if k in negative_keys}
            if len(pred_negative) == 0: continue
            eval_pred(dataset_name,
                      pred_negative,
                      negative=negative,
                      negative_fp_thresh=negative_fp_thresh if negative else None,
                      remove_static = remove_static,
                      static_only = static_only)
    else:
        eval_pred(dataset_name,
                  predictions,
                  remove_static = remove_static,
                  static_only = static_only)

    print("EOF")


def static_dynamic_all_frame_inference(
        file_path="outputs/BUS_BasicConfig_StaticFrame/predictions/breast_ALL@20221108-145033",
        dataset_name="breast_ALL@20221108-145033",
        csv_file="/projects/US/ProjectDatasets/db/breast/pkl_labels_trainval/20221108-145033.csv",
        by_device=False,
        by_hospital=True,
        by_negative=True,
        negative_fp_thresh=0.868509829044342, ):
    # remove_static = True, static_only = False
    for remove_static in (True, False):
        for static_only in (True, False):
            if remove_static is False and static_only is True:
                print("Static Frame Inference")
                main(
                    file_path=file_path,
                    dataset_name=dataset_name,
                    csv_file=csv_file,
                    by_device=by_device,
                    by_hospital=by_hospital,
                    by_negative=by_negative,
                    negative_fp_thresh=negative_fp_thresh,
                    remove_static=remove_static,
                    static_only=static_only
                )
            elif remove_static is True and static_only is False:
                print("Dynamic Frame Inference")
                main(
                    file_path=file_path,
                    dataset_name=dataset_name,
                    csv_file=csv_file,
                    by_device=by_device,
                    by_hospital=by_hospital,
                    by_negative=by_negative,
                    negative_fp_thresh=negative_fp_thresh,
                    remove_static=remove_static,
                    static_only=static_only
                )
            elif remove_static is False and static_only is False:
                print("All Frame Infernece")
                main(
                    file_path=file_path,
                    dataset_name=dataset_name,
                    csv_file=csv_file,
                    by_device=by_device,
                    by_hospital=by_hospital,
                    by_negative=by_negative,
                    negative_fp_thresh=negative_fp_thresh,
                    remove_static=remove_static,
                    static_only=static_only
                )
            else:
                continue
    return


if __name__ == "__main__":
    # main(
    #     file_path="outputs/BUS_BasicConfig_StaticFrame/predictions/breast_ALL@20221108-145033",
    #     dataset_name="breast_ALL@20221108-145033",
    #     csv_file="/projects/US/ProjectDatasets/db/breast/pkl_labels_trainval/20221108-145033.csv",
    #     by_device=False,
    #     by_hospital=True,
    #     by_negative=True,
    #     negative_fp_thresh=0.868509829044342,
    # )
    static_dynamic_all_frame_inference(
        file_path="outputs/BUS_BasicConfig_StaticFrame/predictions/breast_ALL@20221108-145033",
        dataset_name="breast_ALL@20221108-145033",
        csv_file="/projects/US/ProjectDatasets/db/breast/pkl_labels_trainval/20221108-145033.csv",
        by_device=False,
        by_hospital=True,
        by_negative=True,
        negative_fp_thresh=0.868509829044342,
    )

```
</details>

<details>
<summary>ultrasound_vid/evaluation/video_evaluation.py</summary>

```python
import logging
import os
import pickle
from collections import OrderedDict, deque
from itertools import chain
from tempfile import NamedTemporaryFile

import numpy as np
import seaborn as sns
import torch
from detectron2.data import detection_utils as d2utils
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.structures import Instances, Boxes
from detectron2.structures import pairwise_iou
from detectron2.utils import comm
from ultrasound_vid.evaluation.average_recall import average_recall
from ultrasound_vid.evaluation.eval_utils import check_center_cover
from ultrasound_vid.evaluation.mean_ap import eval_map

FP_ID = 0
sns.set_style("darkgrid")


class UltrasoundVideoDetectionEvaluator(DatasetEvaluator):
    """
    Evaluate Pascal VOC AP.
    It contains a synchronization, therefore has to be called from all ranks.

    Note that this is a rewrite of the official Matlab API.
    The results should be similar, but not identical to the one produced by
    the official API.
    """

    def __init__(self, dataset_name, rpn_only=False):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        if isinstance(dataset_name, list):
            self._dataset_name = dataset_name[0]
            self._dataset_list = dataset_name
            self.meta = MetadataCatalog.get(dataset_name[0])
        else:
            self._dataset_name = dataset_name
            self._dataset_list = None
            self.meta = MetadataCatalog.get(dataset_name)
        if self._dataset_list is not None:
            buf = []
            for dataset_name in self._dataset_list:
                buf.append(DatasetCatalog.get(dataset_name))
            dataset_dicts = chain.from_iterable(buf)
        else:
            dataset_dicts = DatasetCatalog.get(self._dataset_name)
        dataset_dicts = {d["relpath"]: d for d in dataset_dicts}
        self.dataset_dicts = dataset_dicts
        self._class_names = self.meta.thing_classes
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._prediction_files = list()
        self._rpn_only = rpn_only
        self.predictions = None

    def process(self, video_info, video_outputs, dump_dir):
        dump_dir = os.path.join(dump_dir, "predictions", self._dataset_name)
        os.makedirs(dump_dir, exist_ok=True)
        relpath = video_info["relpath"]
        video_outputs = [
            o.to(self._cpu_device) if isinstance(o, Instances) else o
            for o in video_outputs
        ]
        save_file = os.path.join(dump_dir, relpath.replace("/", "_") + ".pth")
        torch.save({relpath: video_outputs}, save_file)
        self._prediction_files.append(save_file)

    def load_predictions(self, dump_dir):
        dump_dir = os.path.join(dump_dir, "predictions", self._dataset_name)
        if not comm.is_main_process():
            return
        from glob import glob
        prediction_files = glob(os.path.join(dump_dir, "*.pth"))
        print(f"loaded {len(prediction_files)} pred videos")
        self.predictions = {}
        for f in prediction_files:
            self.predictions.update(torch.load(f, map_location="cpu"))

    def evaluate(self,
                 remove_static = True,
                 static_only = False,
                 dump_dir=None,
                 calc_delay=False,
                 eval_results_only=False,
                 fixed_recall=0.7,
                 fixed_thresh=None):
        if not comm.is_main_process():
            return
        if self.predictions is None:
            assert dump_dir is not None
            self.load_predictions(dump_dir=dump_dir)
        dataset_dicts = self.dataset_dicts
        num_videos = len(dataset_dicts)
        dataset_dicts_temp_file = NamedTemporaryFile().name
        with open(dataset_dicts_temp_file, "wb") as fp:
            pickle.dump(dataset_dicts, fp)
        if self._rpn_only:
            ar = calculate_mmdet_ar(
                self.predictions,
                self._dataset_name,
                self.meta,
                dataset_dicts_temp_file,
                self._logger.info,
            )
            ret = OrderedDict()
            ret["AR"] = {"AR": ar}
            return ret
        (
            mAP,
            eval_results,
            prob_thresh,
            multi_mAP,
            multi_eval_results,
            scale_mAP,
            scale_eval_results,
        ) = calculate_mmdet_ap(
            self.predictions,
            self._dataset_name,
            self.meta,
            dataset_dicts_temp_file,
            self._logger.info,
            fixed_recall=[fixed_recall],
            remove_static = remove_static,
            static_only = static_only
        )

        if dump_dir is not None:
            import json
            eval_results_json = {}
            for k, v in eval_results[0].items():
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                elif isinstance(v, np.float32):
                    v = float(v)
                eval_results_json[k] = v
            json.dump(
                eval_results_json,
                open(os.path.join(dump_dir, f"eval_results_{self._dataset_name}.json"), "w")
            )

        if eval_results_only:
            return eval_results

        prob_thresh = prob_thresh[0][0] if fixed_thresh is None else fixed_thresh
        video_fp = calculate_video_level_fp(
            self.predictions,
            self._dataset_name,
            self.meta,
            dataset_dicts_temp_file,
            iou_thresh=0.5,
            prob_thresh=[prob_thresh],
            _print=self._logger.info,
        )
        ar = calculate_mmdet_ar(
            self.predictions,
            self._dataset_name,
            self.meta,
            dataset_dicts_temp_file,
            self._logger.info,
        )
        ret = OrderedDict()
        ret["Recall"] = {"R@16": ar}
        prec = [np.interp(0.7, x["recall"], x["precision"]) for x in eval_results]
        multi_prec = [
            np.interp(0.7, x["recall"], x["precision"]) for x in multi_eval_results
        ]
        if np.isnan(multi_mAP):
            ret["Precision"] = {"P@R0.7": prec[0], "multi_P@R0.7": float("nan")}
        else:
            ret["Precision"] = {"P@R0.7": prec[0], "multi_P@R0.7": multi_prec[0]}
        recall = scale_eval_results[0]["recall"]
        precision = scale_eval_results[0]["precision"]
        scale_prec = [np.interp(0.7, recall[i], precision[i]) for i in range(5)]
        # print("lesion", [len(recall[i]) for i in range(5)])
        ret["Scale Precision"] = {f"P@R0.7_{i}": scale_prec[i] for i in range(5)}
        ret["Average Precision"] = {"AP50": mAP, "multi_AP50": multi_mAP}
        ret["Scale Average Precision"] = {f"AP50_{i}": scale_mAP[i] for i in range(5)}
        ret["FP stat"] = {"FP/min": np.mean(list(video_fp.values()))}
        if self._dataset_list is not None:
            ret["Stats"] = {"num_videos": num_videos}
        if calc_delay:
            delay_each_class = calculate_video_level_delay(
                self.predictions,
                self._dataset_name,
                self.meta,
                dataset_dicts_temp_file,
                iou_thresh=0.5,
                prob_thresh=[prob_thresh],
                delay_thresh=6,
                _print=self._logger.info,
            )
            ret["delay stat"] = {
                "average delay": np.mean(list(delay_each_class.values()))
            }
        try:
            os.remove(dataset_dicts_temp_file)
            self._logger.info(">>> Dataset dicts temp file removed!")
        except:
            self._logger.info("=== Dataset dicts temp file removing failed!!! ===")
        return ret


def calculate_mmdet_ar(
        predictions, dataset_name, meta, dataset_dicts_temp_file, _print=print
):
    with open(dataset_dicts_temp_file, "rb") as fp:
        dataset_dicts = pickle.load(fp)
    class_names = meta.thing_classes
    assert len(class_names) == 1, "we support one class only"
    _print(f"Evaluating mmdet-style AP on '{dataset_name}' dataset")
    preds, annos = [], []
    dump_anno = {}
    for relpath in predictions.keys():  # iterate video
        frame_preds = predictions[relpath]
        if relpath not in dataset_dicts:
            _print(f">>> {relpath} not in dataset")
            continue
        dump_anno[relpath] = []
        with open(dataset_dicts[relpath]["frame_annos_path"], "rb") as fp:
            frame_annos = pickle.load(fp)
        assert len(frame_preds) == len(frame_annos)
        frame_annos = list(frame_annos.values())
        assert 0 <= (len(frame_annos) - len(frame_preds)) <= 30
        for pred, raw_anno in zip(frame_preds, frame_annos):  # iterate frame
            ignore = raw_anno["ignore"]
            if ignore:
                continue
            anno_image_shape = (raw_anno["height"], raw_anno["width"])
            anno = d2utils.annotations_to_instances(
                raw_anno["annotations"], anno_image_shape
            )
            if pred is None:
                continue
            try:
                pred_boxes = pred.get("proposal_boxes")
            except:
                pred_boxes = pred.get("pred_boxes")
            anno_boxes = anno.get("gt_boxes")
            iou_mat = pairwise_iou(pred_boxes, anno_boxes)
            match_mat = iou_mat > 0.5
            anno.set("matched", match_mat.any(0))
            raw_anno["anno"] = anno
            raw_anno["preds"] = pred
            dump_anno[relpath].append(raw_anno)
            mmdet_pred = []
            for class_id, class_name in enumerate(class_names):
                try:
                    pred_boxes = pred.get("proposal_boxes").tensor.numpy()
                    pred_scores = (
                        pred.get("objectness_logits").float().sigmoid().numpy()
                    )
                except:
                    pred_boxes = pred.get("pred_boxes").tensor.numpy()
                    pred_scores = pred.get("pred_classes").float().sigmoid().numpy()
                if len(pred_boxes) == 0:
                    pred_boxes = np.zeros((0, 5))
                else:
                    pred_boxes = np.hstack((pred_boxes, pred_scores.reshape(-1, 1)))
                mmdet_pred.append(pred_boxes)
            anno_boxes = anno.get("gt_boxes").tensor.numpy()
            mmdet_anno = {
                "bboxes": anno_boxes,
                "labels": anno.get("gt_classes").numpy() + 1,  # note: we should add 1
                "labels_ignore": np.array([]),  # fake ignore to prevent bugs
                "bboxes_ignore": np.zeros((0, 4)),
            }
            preds.append(mmdet_pred)
            annos.append(mmdet_anno)
    # torch.save(dump_anno, f"{dataset_name}_dump_anno.pth")
    ar = average_recall(preds, annos)
    return ar


def calculate_mmdet_ap(
        predictions,
        dataset_name,
        meta,
        dataset_dicts_temp_file,
        _print=print,
        fixed_recall=[
            0.7,
        ],
        remove_static = True,
        static_only = False,
):
    with open(dataset_dicts_temp_file, "rb") as fp:
        dataset_dicts = pickle.load(fp)
    class_names = meta.thing_classes
    _print(f"Evaluating mmdet-style AP on '{dataset_name}' dataset")
    _print(f"The fixed_recall is {fixed_recall}")
    preds, annos = [], []
    multi_preds, multi_annos = [], []
    for relpath in predictions.keys():  # iterate video
        frame_preds = predictions[relpath]
        if relpath not in dataset_dicts:
            _print(f">>> {relpath} not in dataset")
            continue
        with open(dataset_dicts[relpath]["frame_annos_path"], "rb") as fp:
            frame_annos = pickle.load(fp)

        frame_preds = [p for p in frame_preds if p is not None]
        frame_annos = list(frame_annos.values())
        assert 0 <= (len(frame_annos) - len(frame_preds)) <= 30

        num_frames = dataset_dicts[relpath]["video_info"]["num_frames"]
        sample_frame_idx_base = np.arange(0, num_frames).tolist()  # - 3
        if "ignore_frames" in dataset_dicts[relpath]:
            for ignore in dataset_dicts[relpath]["ignore_frames"]:
                sample_frame_idx_base = list(
                    filter(
                        lambda i: i not in np.arange(ignore[0], ignore[1]),
                        sample_frame_idx_base,
                    )
                )

        if remove_static is False and static_only is True:
            # print("Static Frame Inference")
            sample_index_idx_base_rmstatic = list(
                filter(
                    lambda i: i
                              in dataset_dicts[relpath]["video_info"]["static_frames"],
                    sample_frame_idx_base,
                )
            )
        elif remove_static is True and static_only is False:
            # print("Dynamic Frame Inference")
            # remove static frames
            sample_index_idx_base_rmstatic = list(
                filter(
                    lambda i: i
                              not in dataset_dicts[relpath]["video_info"]["static_frames"],
                    sample_frame_idx_base,
                )
            )
        elif remove_static is False and static_only is False:
            # print("All Frame Infernece")
            sample_index_idx_base_rmstatic = sample_frame_idx_base
        else:
            assert "Wrong Arg. remove_static is True and static_only is True"
            return None
        # if(len(sample_index_idx_base_rmstatic)>50):
        sample_frame_idx_base = sample_index_idx_base_rmstatic

        for pred, raw_anno in zip(frame_preds, frame_annos):  # iterate frame
            ignore = raw_anno["ignore"]
            if ignore:
                continue
            if raw_anno["frame_idx"] not in sample_frame_idx_base:
                continue
            anno_image_shape = (raw_anno["height"], raw_anno["width"])
            anno = d2utils.annotations_to_instances(
                raw_anno["annotations"], anno_image_shape
            )
            mmdet_pred = []
            for class_id, class_name in enumerate(class_names):
                temp_pred = pred[pred.get("pred_classes") == class_id]
                pred_boxes = temp_pred.get("pred_boxes").tensor.numpy()
                pred_scores = temp_pred.get("scores").numpy()
                if len(pred_boxes) == 0:
                    pred_boxes = np.zeros((0, 5))
                else:
                    pred_boxes = np.hstack((pred_boxes, pred_scores.reshape(-1, 1)))
                mmdet_pred.append(pred_boxes)
            anno_boxes = anno.get("gt_boxes").tensor.numpy()
            mmdet_anno = {
                "bboxes": anno_boxes,
                "labels": anno.get("gt_classes").numpy() + 1,  # note: we should add 1
                "labels_ignore": np.array([]),  # fake ignore to prevent bugs
                "bboxes_ignore": np.zeros((0, 4)),
            }
            preds.append(mmdet_pred)
            annos.append(mmdet_anno)
            if anno_boxes.shape[0] > 1:
                multi_preds.append(mmdet_pred)
                multi_annos.append(mmdet_anno)
    mAP, eval_results, prob_thresh = eval_map(preds, annos, fixed_recall=fixed_recall)
    multi_mAP, multi_eval_results, _ = eval_map(
        multi_preds, multi_annos, fixed_recall=fixed_recall
    )
    edges = [
        10,
        103.62432146943111,
        145.9314907756376,
        204.68268124098825,
        311.0434053311531,
        float("inf"),
    ]
    scale_ranges = [(edges[i], edges[i + 1]) for i in range(5)]
    scale_mAP, scale_eval_results, _ = eval_map(
        preds, annos, scale_ranges=scale_ranges, fixed_recall=fixed_recall
    )
    return (
        mAP,
        eval_results,
        prob_thresh,
        multi_mAP,
        multi_eval_results,
        scale_mAP,
        scale_eval_results,
    )


def calculate_video_level_fp(
        predictions,
        dataset_name,
        meta,
        dataset_dicts_temp_file,
        iou_thresh=0.5,
        prob_thresh=None,
        history_length=5,
        _print=print,
        tp_ratio=0.9,
):
    class BoxNode:
        def __init__(self, box, prev=None, istp=False, visited=False, fp_id=-1):
            self.box = box
            self.prev = prev
            self.istp = istp
            self.visited = visited
            self.fp_id = fp_id

    def fill_fp_id(node):
        global FP_ID
        temp = node
        buf = []
        while True:
            buf.append(temp)
            if temp.visited:
                final_fp_id = temp.fp_id
                break
            if temp.prev is None:
                final_fp_id = FP_ID
                FP_ID += 1
                break
            temp = temp.prev
        for item in buf:
            item.visited = True
            item.fp_id = final_fp_id

    # per video
    def mark_tp_preds(preds, annos):
        marked_box_nodes = []
        for _pred, _anno in zip(preds, annos):
            match_mat = check_center_cover(
                _pred.get("pred_boxes"), _anno.get("gt_boxes")
            )
            buf = []
            for k, istp in enumerate(match_mat.any(axis=1)):
                node = BoxNode(_pred.get("pred_boxes")[k], istp=istp.item())
                buf.append(node)
            marked_box_nodes.append(buf)
        return marked_box_nodes

    def merge_nodes(box_nodes):
        hist = deque(maxlen=history_length)
        # forward
        for nodes in box_nodes:
            if len(hist) > 0:
                flatten_hist_nodes = [n for hist_n in hist for n in hist_n]
                hist_boxes = [n.box for n in flatten_hist_nodes]
                pred_boxes = [n.box for n in nodes]
                if len(hist_boxes) > 0 and len(pred_boxes) > 0:
                    hist_boxes = Boxes.cat(hist_boxes)
                    pred_boxes = Boxes.cat(pred_boxes)
                    ious = pairwise_iou(pred_boxes, hist_boxes)
                    match_mat = ious > iou_thresh
                    istp = match_mat.any(dim=1)
                    for k in range(len(istp)):
                        if istp[k]:
                            prev_idx = ious[k].argmax()
                            nodes[k].prev = flatten_hist_nodes[prev_idx]
            hist.append(nodes)
        # backward
        for nodes in box_nodes[::-1]:
            for n in nodes:
                fill_fp_id(n)

    def select_fp_preds(box_nodes):
        global FP_ID
        fps = []
        for _id in range(FP_ID):
            buf = [n for nodes in box_nodes for n in nodes if n.fp_id == _id]
            seq_len = len(buf)
            tp_len = sum([n.istp for n in buf])
            if tp_len / seq_len > tp_ratio:
                continue
            fps.append(buf)
        return fps

    with open(dataset_dicts_temp_file, "rb") as fp:
        dataset_dicts = pickle.load(fp)
    class_names = meta.thing_classes
    single_class_flag = False
    if len(class_names) == 1:
        single_class_flag = True
    if prob_thresh is None:
        prob_thresh = [0.5] * len(class_names)
    _print(f"Calculating FP rate on '{dataset_name}' dataset")
    _print(f"Prob threshold is {prob_thresh}")
    video_fp_rate = {}
    for class_id, class_name in enumerate(class_names):  # iterate class
        fp_cnt = 0
        frame_cnt = 0
        for relpath in predictions.keys():  # iterate videos
            frame_preds = predictions[relpath]
            if relpath not in dataset_dicts:
                _print(f">>> {relpath} not in dataset")
                continue
            with open(dataset_dicts[relpath]["frame_annos_path"], "rb") as fp:
                frame_annos = pickle.load(fp)

            frame_preds = [p for p in frame_preds if p is not None]
            frame_annos = list(frame_annos.values())
            assert 0 <= (len(frame_annos) - len(frame_preds)) <= 30
            preds, annos = [], []

            num_frames = dataset_dicts[relpath]["video_info"]["num_frames"]
            sample_frame_idx_base = np.arange(0, num_frames).tolist()
            if "ignore_frames" in dataset_dicts[relpath]:
                for ignore in dataset_dicts[relpath]["ignore_frames"]:
                    sample_frame_idx_base = list(
                        filter(
                            lambda i: i not in np.arange(ignore[0], ignore[1]),
                            sample_frame_idx_base,
                        )
                    )
            # remove static frames
            sample_index_idx_base_rmstatic = list(
                filter(
                    lambda i: i
                              not in dataset_dicts[relpath]["video_info"]["static_frames"],
                    sample_frame_idx_base,
                )
            )
            sample_frame_idx_base = sample_index_idx_base_rmstatic

            for pred, raw_anno in zip(frame_preds, frame_annos):
                ignore = raw_anno["ignore"]
                if ignore:
                    continue
                if raw_anno["frame_idx"] not in sample_frame_idx_base:
                    continue
                anno_image_shape = (raw_anno["height"], raw_anno["width"])
                anno = d2utils.annotations_to_instances(
                    raw_anno["annotations"], anno_image_shape
                )
                if not single_class_flag:
                    pred = pred[pred.get("pred_classes") == class_id]
                    anno = anno[anno.get("gt_classes") == class_id]

                # ignore low probability predictions
                pred = pred[pred.get("scores") > prob_thresh[class_id]]
                preds.append(pred)
                annos.append(anno)
            marked_box_nodes = mark_tp_preds(preds, annos)
            merge_nodes(marked_box_nodes)

            fps = select_fp_preds(marked_box_nodes)

            fp_cnt += len(fps)
            frame_cnt += min(len(frame_preds), len(frame_annos))
            global FP_ID
            FP_ID = 0
        video_fp_rate[class_id] = fp_cnt / frame_cnt * 1800
    return video_fp_rate


def calculate_frame_level_fp_fn(single_video_preds, single_video_annos, iou_thresh):
    fps, fns = [], []
    anno_idx = 0
    for pred in single_video_preds:
        if pred is None:
            fps.append(0)
            fns.append(0)
            continue
        anno = single_video_annos[anno_idx]
        ious = pairwise_iou(pred.get("pred_boxes"), anno.get("gt_boxes"))
        match_mat = ious > iou_thresh  # #predx#anno
        num_fp = (~match_mat.any(dim=1)).sum().item()
        num_fn = (~match_mat.any(dim=0)).sum().item()
        fps.append(num_fp)
        fns.append(num_fn)
        anno_idx += 1
    return np.array(fps), np.array(fns)


def calculate_video_level_delay(
        predictions,
        dataset_name,
        meta,
        dataset_dicts_temp_file,
        iou_thresh=0.5,
        prob_thresh=None,
        delay_thresh=10,
        _print=print,
):
    with open(dataset_dicts_temp_file, "rb") as fp:
        dataset_dicts = pickle.load(fp)
    class_names = meta.thing_classes
    if prob_thresh is None:
        prob_thresh = [0.5] * len(class_names)

    _print(f"Calculating delay on '{dataset_name}' dataset")
    average_delay = {}
    for class_id, class_name in enumerate(class_names):  # iterate class
        delay_cnt = 0
        total_unique_object_number = 0
        for relpath in predictions.keys():  # iterate videos
            frame_preds = predictions[relpath]
            if relpath not in dataset_dicts:
                _print(f">>> {relpath} not in dataset")
                continue
            with open(dataset_dicts[relpath]["frame_annos_path"], "rb") as fp:
                frame_annos = pickle.load(fp)

            frame_preds = [p for p in frame_preds if p is not None]
            frame_annos = list(frame_annos.values())
            assert 0 <= (len(frame_annos) - len(frame_preds)) <= 30
            preds, annos = [], []
            for pred, raw_anno in zip(frame_preds, frame_annos):
                ignore = raw_anno["ignore"]
                if ignore:
                    continue
                anno_image_shape = (raw_anno["height"], raw_anno["width"])
                anno = d2utils.annotations_to_instances(
                    raw_anno["annotations"], anno_image_shape
                )

                pred = pred[pred.get("pred_classes") == class_id]
                anno = anno[anno.get("gt_classes") == class_id]

                # ignore low probability predictions
                pred = pred[pred.get("scores") > prob_thresh[class_id]]
                preds.append(pred)
                annos.append(anno)
            # import pdb
            # pdb.set_trace()
            box_delay_list = []
            for frame_id in range(len(preds)):
                if (frame_id == 0 and len(annos[frame_id]) != 0) or (
                        frame_id != 0
                        and len(annos[frame_id]) != 0
                        and len(annos[frame_id - 1]) == 0
                ):
                    box_delay_list.append([annos[frame_id], 0])
                    total_unique_object_number += len(annos[frame_id])
                if len(box_delay_list) != 0:
                    if len(preds[frame_id]) != 0:
                        for box_id, box in enumerate(box_delay_list):
                            ious = pairwise_iou(
                                preds[frame_id].get("pred_boxes"),
                                box[0].get("gt_boxes"),
                            )
                            if torch.max(ious) > iou_thresh:
                                delay_cnt += box[1]
                                box_delay_list[box_id][1] = -1
                            else:
                                box_delay_list[box_id][1] += 1
                        for box_id, box in enumerate(box_delay_list):
                            if box[1] >= delay_thresh:
                                delay_cnt += box[1]
                                box_delay_list[box_id][1] = -1
                        box_delay_list = [
                            item for item in box_delay_list if item[1] >= 0
                        ]
                    else:
                        for box_id, box in enumerate(box_delay_list):
                            box_delay_list[box_id][1] += 1
                        for box_id, box in enumerate(box_delay_list):
                            if box[1] >= delay_thresh:
                                delay_cnt += box[1]
                                box_delay_list[box_id][1] = -1
                        box_delay_list = [
                            item for item in box_delay_list if item[1] >= 0
                        ]
            for box_id, box in enumerate(box_delay_list):
                delay_cnt += box[1]
        if total_unique_object_number == 0:
            _print(f"All labels are negative. No delay.")
            average_delay[class_id] = -1
        else:
            average_delay[class_id] = delay_cnt / total_unique_object_number
            _print(f"average delay is: {delay_cnt / total_unique_object_number}")

    return average_delay

```
</details>

使用``remove_static``和``static_only``两个形参对以上模式进行编码，规则如下

|             |          |           |
|-------------|----------|-----------|
| **R_S\S_O** | **True** | **False** |
| **True**    | Null     | 运动帧       |
| **False**   | 静止帧      | 全部帧       |

在BUS数据集上，以0.0概率移除静止帧（不移除）实验结果

| **北京大学肿瘤医院_阳性** | **AP50** | **P@0.7** | **FP** | **R@16** |
|:---------------:|:--------:|:---------:|:------:|:--------:|
| **Static**      | 92.81    | 98.42     | 0.0658 | 94.42    |
| **Dynamic**     | 86.91    | 92.94     | 0.5265 | 94.42    |
| **All**         | 91.20    | 98.38     | 0.0658 | 94.42    |



### 1.4. 下周工作

1. 以0.2, 0.4, 0.6, 0.8, 1.0的移除静止帧概率验证TUS上的表现；（静止/运动/全部）

## 2. Ultrasound-VID项目学习笔记

[Repo](https://github.com/xjtulyc/ultrasound_vid_docs)