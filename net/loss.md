<!--
 * @Author: jhq
 * @Date: 2025-02-08 14:16:10
 * @LastEditTime: 2025-03-07 18:54:36
 * @Description: 
-->
* focal loss：为解决正负样本不均衡设计，降低容易分类的样本的权重，FL(p,q) = -Σ(1-p_x)^gamma*log(q_x)
* quality focal loss: QFL(σ) = -|y-σ|^β*((1-y)log(1-σ)+y*log(σ))
* distribution focal loss: DFL(Si,Si+1) = -((yi+1 - y)*log(Si)+(y - yi)*log(Si+1)), Si=(yi+1 - y)/(yi+1 - yi), Si+1=(y - yi)/(yi+1 - yi)
* MSE-Mean Square Error,均方差损失-二次损失-L2损失
    - MSE = 1/n*Σ(y_i-y_i^)^2
* CTC
* 目标检测损失函数，多任务损失 Lp,u,tu,v=Lcls(p,u)+λ[u≥1]Lloc(t^u,v)
* Balanced L1 Loss,平衡L1损失

* L1损失-绝对值损失，1/n*Σ|y_i-y_i^|
* L2损失-均方差损失，1/n*Σ(y_i-y_i^)^2
* CE-cross entropy-交叉熵损失:CE(p,q) = - Σp_x*log(q_x),p-真实值，q-预测值
    - 单标签多分类
    - 处理多选一问题，
    - 输出层激活函数：softmax，每个样本一个概率分布，所有类别概率和为1
* BCE-binary cross entropy-二分类交叉熵损失:BCE(p,q) = -p_x*log(q_x)+(1-p_x)*log(1-q_x)
    - 二分类/多个独立标签
    - 输出层激活函数：sigmoid，每个样本一个概率，0-1之间