<!--
 * @Author: jhq
 * @Date: 2022-11-27 18:54:10
 * @LastEditTime: 2022-12-05 22:17:21
 * @Description: 
-->
#### 改进
    * 基于RepVGG style 设计了可重参数化、更高效的骨干网络 EfficientRep Backbone和Rep-PAN Neck
    * 优化设计Efficient Decoupled Head,降低一般解耦头的额外延时开销并维持精度
    * 采用Anchor-free无锚范式， 同时辅以SimOTA标签分配策略以及SIoU边界回归损失