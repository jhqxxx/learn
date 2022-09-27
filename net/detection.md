### 目标检测基础
#### anchors理解
    人为预设好的不同大小，不同长宽比的参照框
#### IoU计算
    IoU: 两框交集/两框并集
#### NMS计算
    step1:将所有检测出来的bbox按class score划分
    step2:在每个集合内根据各个bbox的class score做降序排列，得到list_k
    step3:从list_k中的第一个bbox开始，计算bbox_x与lisk_k中其他bbox_y的IoU,若IoU大于设定阈值，则剔除bbox_y,保留bbox_x，从list_k中取出
    step4:重复step3中的迭代操作，直至list_k中所有的bbox都完成筛选
    step5:对每个集合的list_k,重复step3,step4的迭代操作，直至所有list_k都完成筛选
#### Soft NMS
    以一个权重的方式，将获得的IOU取高斯指数后乘上原得分，之后重新排序再继续循环
#### 模型评价标准 AP与mAP
    precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    根据检测框与真实框对比，IoU/置信度，计算样本数据的precision和recall
    绘制PR曲线, 将PR曲线平滑处理
    AP = ∫p(r)dr，  r->0-1
    针对自变量r(recall),对曲线p(precision)做积分
    不同IoU阈值可以得到多个AP值，如AP50，AP75等
    mAP是在所有类别下的均值
### 二阶段
#### Faster RCNN
#### Mask RCNN
#### Cascade RCNN

### 一阶段
#### Yolo系列
#### SSD
#### Retinanet

### 无anchor
#### centernet