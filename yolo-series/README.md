<!--
 * @Author: jhq
 * @Date: 2025-01-03 16:52:22
 * @LastEditTime: 2025-03-10 13:19:25
 * @Description: 

-->
#### detect
使用猫狗数据集测试
60个类别,每个类别150张图

###### anchor-free
* 检测头直接预测目标的位置和大小
    - head输出多尺度的特征图
        - 每个点只预测一个框
        - 预测特征图的中心点或关键点来生成检测框
        - 特征图每个位置直接预测边界框的四个值
    - 中心点预测
        - 预测边界框的中心点偏移-相对于网格中心的偏移
    - 直接预测边界框的坐标
        - 边界框宽高-直接预测绝对尺寸
    - 预测值（dx,dy, w, h）
        - 中心点坐标：（x_grid+dx, y_grid+dy）
        - 宽高： w,h
* 解耦检测头/Decoupled Head
    - 分类分支：预测物体的类别概率
        - 分类损失：BCELoss/FocalLoss        
    - 回归分支：预测边界框的中心点偏移和宽高
        - 回归损失：CIoU/L1Loss-早期
    - 置信度分支/可选：预测是否为物体的置信度，部分模型将其合并到分类中
        - BCELoss
* 正负样本分配策略
    - 如何确定哪些预测框负责学习真实框-正样本分配
    - 根据预测框和真实框的匹配程度来分配正样本
    - SimOTA
        - 初步筛选候选框：选择与真实框中心点最近的若干预测框。如中心点距离<阈值
        - 计算loss矩阵:对每个候选框，计算分类损失+回归损失的总代价
        - 为每个真实框分配总代价最小的前k个预测框，k根据真实框 的大小动态调整
    - 动态分配
* 分类损失
    - GIoU
    - DIoU
    - CIoU = IoU - ρ^2(b_pred, b_gt)/c^2 - alpha*v
        - ρ:预测框与真实框中心点的欧式距离
        - c:最小包围框的对角线长度
        - v:宽高比的一致性度量        
* 优势
    - 减少超参数依赖
    - 简化模型
    - 减少计算量
    - 提高检测速度
    - 提升准确率

###### anchor-based
* 需要手动设计Anchor的尺寸-如宽高比，对数据集敏感，泛化性差
* 引入大量计算-每个位置预测多个Anchor框，效率低

##### 12
* A2C2f
* ABlock
* AAttn

##### yolov11-n
* 参数量减少，计算量降低，采用更高效的模块设计
    - DualConv
    - 组卷积
    - 异构卷积
    - 引入空间注意力机制，增强了对小物体和遮挡的处理能力
* train: mAP50-0.846,mAP50:95-0.766
* time: 4.5 hours
* infer:有个奇怪的现象,同一张图单独推理和放在多张图组成list时的推理结果不一样,待发现原因！！！！
* 网络结构
    - C2PSA
        - v11自定义模块,带有注意力机制
        - 使用一系列PSABlock模块
    - PSABlock
    - C3K2
        - 继承自C2f,引入基于C3k模块的瓶颈层
    - C3k
    - SPPF
    - 解耦头的分类检测头增加DWConv
    - Split
    - 11Detect 
    - Conv
    - Upsample
    - Bottleneck
    - shortcut

* AGLU
* DFL:
* Proto:包含转置卷积的上采样模块
* HGStem: PPHGNetV2的分支结构StemBlock,5卷积,1最大池化
* HGBlock: PPHGNetV2的分支结构,2卷积,1LightConv,squeeze+excitation???
* SPP:Spatial Pyramid Pooling空间金字塔池化,对输入特征图进行池化操作,将特征图进行上采样,再与原特征图进行concat操作,实现特征图上采样。有三个池化层k=(5,9,13),输入特征图分别通过三个池化层
* SPPF:SPP Fast,一个池化层k=5,输入特征图通过该池化层三次,每次的结果记录并进行下一个池化,经过两次k=5的池化效果与一次k=9的池化一致,经过三次k=5的池化效果与一次k=13的池化一致。
- SPPF相比SPP,由于ksize小些,确实会起到速度提升的效果
* C1:CSP Bottleneck with 1 convolution
* C2: CSP Bottleneck with 2 convolutions 
* C2f: C2 Fast
- 但是C2f相比C2f, BottleNeck的个数一致,channel一致,Ksize一致,Mapsize也一致,Flops算下来是一样的,不明白为什么C2f比C2快？？？？？
* C3:CSP Bottleneck with 3 convolutions
* Bottleneck: 有shortcut的add操作
* BottleneckCSP: cross stage partial
* C3x: C3 with cross-convolutions,但是源码没有写完这个模块,,只有一个init,都没有forward。
* cross-convolutions: 跨层卷积,通过跨通道或跨空间位置的方式进行卷积操作,以增强模型的特征提取能力和表达能力。
* RepC3:Bottleneck是RepConv
* C3TR: C3 with Transformer block,源码也没有写完这个模块,bottleneck是TransformerBlock.
* C3Ghost: C3 with GhostBottleneck,源码也没有写完这个模块, bottleneck是GhostBottleneck.
* GhostBottleneck: 由GhostConv和DWconv组成.
- GhostNet: 轻量级卷积网络, 主流CNN的中间特征图中存在大量的冗余,但是冗余的特征图又是必要的,如果Cout-m太小,网络宽度不够,预测结果也不好。GhostNet提出减少产生冗余特征图的计算消耗,即减小Cout-n,n<=m,然后使用k=m/n个线性计算,将输入特征图的Cout扩张到m。
* GhostConv
* DWConv
* ResNetBlock:有残差连接,add操作
* ResNetLayer:多个ResNetBlock堆叠
* MaxSigmoidAttnBlock: 应用attention模块。
* C2fAttn: C2f with an additional MaxSigmoidAttnBlock 
* ImagePoolingAttn: Enhance the text embeddings with image-aware information,对输入进行pool操作
* ContrastiveHead: ???? 对比学习？？？用于计算区域-文本相似度,初始化时设置了一个偏置项和一个缩放因子,前向传播过程中,对输入特征进行归一化处理,然后通过矩阵乘法计算相似度,并应用缩放因子和偏置项
* BNConstrastiveHead:使用batch norm 代替L2范式
* RepNCSPELAN4:
* ELAN1
* AConv: average pooling + conv
* ADown: downsampling with average pooling and maxpooling
* SPPELAN: 
* CBLinear:用1*1卷积模拟线性层
* CBFuse: 特征融合, 插值运算再相加,没懂有什么意义？？？
* C3f: Faster Implementation of CSP Bottleneck with 2 convolutions
* C3k2: C3f optional C3k block
* C3k:
* RepVGGDW: RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture
* CIB: conditional identity block
* C2fCIB: C2f with CIB
* Attention:使用conv达到线性-linear计算qkv的效果
* PSABlock: Position-Sensitive Attention block
* PSA: 比PSABlock多两个conv
* C2PSA: PSA模块代替Bottleneck
* C2fPSA: PSABlock代替Bottleneck
* SCDown: downsampling with separable convolutions

* Conv: act+bn+conv
* Conv2: simplified RepConv with Conv fusing
* LightConv: 使用1*1卷积降低通道数,再使用DWconv
* DWConv: depthwise convolutions,下采样
* DWConvTranspose2d: depthwise convolutions,上采样
* ConvTranspose:act+bn+ConvTranspose2d,上采样
* focus:input shape (b,c,w,h)->(b,c*4,w/2,h/2)???不知道干啥用
* GhostConv:
* RepConv: 融合bn+conv3*3+conv1*1
* ChannelAttention: channel维度的注意力机制
* SpatialAttention: spatial dimension attention
* CBAM: channel attention + spatial attention
* Concat: Concatenate a list of tensors along dimension

###### 任务
* 分类
* 检测
* 分割
* 跟踪
* 关键点
* 旋转目标检测

##### yolov10-n
* 清华发布,引入无NMS训练策略
* train: mAP50-0.844,mAP50:95-0.765
* time: 7.1 hours
* 网络结构
    - PSA:部分自注意力
    - MSHA:自注意力模块
    - SCDown
    - C2fCIB
    - CIB
    - 双标签分配

##### yolos
* 网络结构
    - TAL-任务分配学习


##### yolov9-t
* 引入可编程梯度信息（PGI）优化技术
* 通用高效层聚合网络GELEN
* 可逆函数
* t类似其他的n
* train: mAP50-0.842,mAP50:95-0.762
* time: 7.2 hours

##### yolov8 
任务范围增加,新增实例分割、关键点检测、分类等,工程能力更强
* 网络结构
    - 引入transformer和高效的注意力机制 
    - C2f
    - 解耦头？？？
    - anchor-free？？？

##### yolov7
* 模型结构重参化,repVGG也是一种
* 动态标签分配
* ELAN
* ELAN-W
* SPPCSPC
* EMA
* 刚开始有anchor，后面更新了无anchor版本

##### yolov6 
* 美团
* 网络结构
    - EfficientRep: repVGG训练时多分支,部署时可以等效融合为单个3x3卷积的可重参数化结构
    - Rep-PAN
    - 解耦检测头
    - SIoU
    - 无anchor
    - SiLU

##### yoloX
* 网络结构
    - 无anchor

##### yolov5 
###### 输出  
* 工程化贡献更大,使用Pytorch重写,提供多个规模模型,网络本身改良点不多
* 网络结构
    - Focus-切片操作:608*608*3->304*304*12
    - 自适应锚框计算
    - 自适应图片缩放
    - YOLOv5 将输入图像划分为 S×S 的网格（例如 3 个不同尺度的特征图：80×80、40×40、20×20）
    - 每个网格会预设一组 Anchor 框（不同尺度的参考框，例如 YOLOv5 默认每个尺度有 3 个 Anchor
    - Anchor 的作用：提供不同形状的“候选框模板”，帮助模型预测不同大小的物体（例如小物体用小的 Anchor，大物体用大的 Anchor
    - 对每个真实框（Ground Truth），YOLOv5 会找到 最匹配的 Anchor 和 对应的网格位置
        - Step 1：计算真实框与所有 Anchor 的宽高比匹配度,选择形状最接近的 Anchor
        - Step 2：确定真实框的中心点所属的网格, 定位到具体的网格位置
    - 扩展匹配：跨网格与跨层策略
        - 跨网格匹配：真实框中心点所在的网格及其 邻近网格（例如上下左右）也可能参与匹配
        - 跨层匹配：真实框可能在多个尺度的特征图上匹配（例如小物体在细粒度高的特征图匹配，大物体在粗粒度特征图匹配）
        - 让更多预测框参与学习，提高模型鲁棒性
    - 正样本筛选：基于 IoU 或距离的阈值
        - 条件 1：预测框与真实框的 IoU > 阈值（例如 0.2）,IoU（交并比）衡量预测框与真实框的重叠程度
        - 条件 2：预测框中心点与真实框中心点的距离 < 阈值,避免匹配到距离过远的预测框。
    - 每个真实框会被分配给 1~3 个预测框（具体数量由超参数控制）
    - 计算损失：
        - 正样本：
            - 计算 边界框回归损失（例如 CIoU Loss，优化位置和大小）
            - 计算 分类损失（BCE Loss，优化类别概率）
            - 计算 目标置信度损失（BCE Loss，置信度趋近于 1）
        - 负样本：
            - 计算 目标置信度损失（置信度趋近于 0）


##### yolov4
* 网络结构
    - 最小组件:conv+bn+mish
    - CSPDarknet53:CSP模块,将输入特征图分为两半,分别进行卷积操作,再进行相加操作,减少计算量,保证准确率。
    - SPP:使用k={1*1, 5*5, 9*9, 13*13}的最大池化,padding=ks//2,再将不同尺度的特征图进行concat操作。
    - FPN+PAN:FPN:自顶向下,将小尺度特征图上采样,再与大尺度特征图concat,PAN:将FPN操作后的大尺度特征图再进行下采样,与对应尺度的FPN结果进行concat得到用于检测的特征图。
    - Mish激活函数
    - Dropblock:随机丢弃一定比例的输入特征图-在特征图上部分区域数据置为0,减少过拟合,提高泛化能力。dropout主要作用与全连接层,dropblock可以作用在任何卷积层上。
    - 输出特征图的尺寸为:19*19*255,38*38*255,76*76*255,
* 使用Mosaic增强
    - 把4张图片，通过随机缩放、随机裁剪、随机排布的方式进行拼接
    - 丰富了检测数据集，特别是缩放增加了很多小目标
* cmBN
* SAT自对抗训练
* CIoU损失函数
* DIOU_nms


##### yolov3
* 网络结构:
    - 最小组件:conv+bn+LeakyReLU
    - res残差结构:残差结构,将输入特征图与中间层进行相加,add操作
    - upsample
    - concat:将中间层与后面某一层的上采样进行拼接,会扩充维度
    - 没有池化层,特征图尺寸变小通过改变卷积计算的步长来实现
    - 输出特征图的尺寸为:13*13*255,26*26*255,52*52*255,借鉴FPN(feature pyramid network特征金字塔)采用多尺度来对不同大小的目标进行检测,小尺寸的负责大目标,大尺寸的负责小目标。
    - 每个网格单元预测3个box,每个box有x,y,w,h,confidence,class_probability即5+类别个数,coco一共有80个类别,即3*(5+80)=255

* 单尺度->多尺度:引入多个特征层来改善对小物体的检测能力


##### yolov2
* 网络结构:
    - batch normalization
    - 加入anchor机制,但是没有采用
    - 直接预测相对位置,预测框中心点相对于网格单元左上角的相对坐标

##### yolov1
* 网络结构
    - conv2d
    - maxpool2d
    - leakyrelu
    - input:448*448*3
    - output: 7*7*30(20 + 5 * 2): 20个类别 + 两个框,每个框:x,y,w,h,score
    - 将输入图片划分为7*7个网格,网格与网络的输出7*7对应,标注框中心点在网格内,则该网格的输出为1,否则为0,每个网格只能预测两个框,且只有20个类别。
* 损失函数:考虑预测框的位置、宽高、是否有框、种类、置信度

#### point



#### segmentation

#### classification


###### 池化
* 平均池化
* 最大池化
* K-max池化

###### 参数量与FLOPs计算
* conv2d参数量: weights + bias: Cin * Cout * K * K + Cout
* conv2d FLOPs: Param_conv2d * Map_outh * Map_outw: Map_outh和Map_outw为输出特征图的宽高
* BatchNorm2D 参数量: 4 * Cout:β,γ,μ,σ^2
* BatchNorm2D FLOPs: 2 * Cout * Map_outh * Map_outw
* 线性层计算量:Cin * Cout + Cout
* 线性层FLOPs:Cin * Cout

###### SqueezeNet
* 将3*3卷积替换成1*1卷积,减少参数量
* 减少3*3卷积的通道数,减少参数量
* Squeezeblock:一组连续的1*1卷积
* expandBlock:一组连续的1*1卷积和3*3卷积
###### SqueezeNext
* 将3*3卷积分解为3*1卷积和1*3卷积,参数量由k^2降为2*k,
* 将expandBlock中1*1卷积移除
* 引入ResNet的shortcut结构
* 文章提及深度可分离卷积在硬件上的运行效果不好

###### Attention
* 可以理解为对输入的每个数据赋予一个权重,然后根据权重对输入进行加权求和
* Seq2Seq:基于一个Encoder 和一个Decoder来构建EndtoEnd模型,Encoder把输入X编码成一个固定长度的隐向量Z,Decoder基于隐向量Z解码出目标输出Y,存在问题:
    - 固定长度的隐向量Z,忽略了输入X的长度
    - 对输入进行编码时,对x中的每个数据都赋予相同的权重,没有区分度
* 传统Attention Mechanism:基于source端和target端的隐变量计算attention,得到的结果是source端的每个词与目标端每个词之间的依赖关系。
* self Attention:分别在source端和target端进行,捕捉source端或target端自身的词之间的依赖关系。
* Transformer:基于self Attention
    - encoder和decoder结构由多个堆叠的Multi-Head Attention单元组成
    - Multi-Head Attention:由多个Scaled Dot-Product Attention组成
    - Scaled Dot-Product Attention:首先把输入input经过线性变换(input分别乘Wq,Wk,Wv)得到Q,K,V,然后把Q和K做点积,得到输入input词与词之间的依赖关系,再经过尺度变换scale,掩码mask(decoder有)和softmax操作,得到attention权重,最后把attention权重和V做点积,得到输出。尺度变换是为了防止输入值过大导致训练不稳定,mask则是为了保证时间的先后关系,softmax将结果归一化为概率分布。其中的线性变换Wq,Wk,Wv一般是nn.Linear的参数。
    - Attention(Q,K,V) = softmax(QK^T/sqrt(dk))V
    - Multi-Head就是可以有不同的Q,K,V进行计算,最后将结果结合起来。
    - 对于encoder来说,Q,K,V均来自前一层encoder的输出。
    - 对于decoder来说,Q,K,V不仅接受前一层的输出,还接受来自encoder的输出,且要进行mask操作:我们只能拿到前面已经翻译过的输出的词语。


###### 蒙特卡洛？？？？


-- 关闭wandb disabled