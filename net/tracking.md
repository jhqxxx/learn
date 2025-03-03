<!--
 * @Author: jhq
 * @Date: 2025-02-08 14:11:35
 * @LastEditTime: 2025-03-03 10:31:23
 * @Description:
-->

###### 卡尔曼滤波 kalman filter

- 应用在跟踪上，有 8 个状态:(x, y, a, h, vx, vy, va, vh)
- 预测的状态是一个分布，均值为 μ， 方差为 σ
- 状态之间具有一定的相关关系，使用协方差矩阵来表示
- 用匀速运动来估计下一个状态的分布
  - 假设预测状态 P=[position, velocity]
  - pk = pk-1 + delta_t \* vk-1
  - vk = vk-1
  - 可以写成矩阵形式
  - xk = Fk \* xk-1
  - 协方差矩阵具有以下的变换恒等式
    - Cov(x) = Σ
    - Conv(Ax) = AΣA^T
  - 预测的协方差矩阵为：Σk = Fk _ Σk-1 _ Fk^T
- 当有外部因素，可以增加一个向量 uk,作为对系统的修正
  - 如果有匀加速度，则预测状态为：
  - pk = pk-1 + delta_t * vk-1 +1/2*a\*delta_t^2
  - vk = vk-1 + a\*delta_t
  - 写成矩阵形式
  - xk = Fk * xk-1 + Bk*uk
  - Bk 被称作控制矩阵，uk 被称作控制向量
  - 外部不确定性，用 Q 矩阵来表示，Q 矩阵表示了预测的噪声
  - 预测的协方差矩阵为：Σk = Fk _ Σk-1 _ Fk^T + Qk
- 测量值也有一个概率分布
- 可以将两个概率分布相乘得到新的分布 ？？？？
- 先通过上一轮时刻的状态进行估计，会得到对应的一个预测状态
- 然后我们会获得当前状态传感器的读数，也会获得一个状态
- 然后我们将两个状态的估计进行融合，更新得到最后的最佳状态分布？？？？
- 融合的不理解



1，找一篇最新 tracking survey，看跟踪分为多少方法？怎么划分的？最新技术是什么？（Siamese network 是最新较为主流的方法，但是也算黔驴技穷了。感觉最新的 paper 也没啥新意，都是在小打小闹，堆数据）；

2. 看最新的 tracking paper，有人整理了 tracking list，主要是发表在顶会上的一些方法。具体可以去 GitHub 搜索。

3. 最重要的当然是跑代码，看哪些 tracker，在哪些 video 上跟踪失效。分析为啥失效，怎么才能跟得上，你不就有自己的 idea 了吗？然后 paper 就在路上了。

4. 看看最新有啥新技术，可以和 tracking 结合的。比如，transformer？self-supervised learning？Multi-modal？Local-Global-search？总有一款适合你。
