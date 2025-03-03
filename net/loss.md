<!--
 * @Author: jhq
 * @Date: 2025-02-08 14:16:10
 * @LastEditTime: 2025-03-01 13:15:23
 * @Description: 
-->
* cross entropy-交叉熵：CE(p,q) = - Σp_x*log(q_x),p-真实值，q-预测值
* focal loss：降低容易分类的样本的权重，FL(p,q) = -Σ(1-p_x)^gamma*log(q_x)
* quality focal loss: QFL(σ) = -|y-σ|^β*((1-y)log(1-σ)+y*log(σ))
* distribution focal loss: DFL(Si,Si+1) = -((yi+1 - y)*log(Si)+(y - yi)*log(Si+1)), Si=(yi+1 - y)/(yi+1 - yi), Si+1=(y - yi)/(yi+1 - yi)
* MSE-Mean Square Error,均方差损失-二次损失-L2损失
    - MSE = 1/n*Σ(y_i-y_i^)^2
* CTC
* 目标检测损失函数，多任务损失 Lp,u,tu,v=Lcls(p,u)+λ[u≥1]Lloc(t^u,v)
* Balanced L1 Loss,平衡L1损失