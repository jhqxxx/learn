* cross entropy-交叉熵：CE(p,q) = - Σp_x*log(q_x),p-真实值，q-预测值
* focal loss：降低容易分类的样本的权重，FL(p,q) = -Σ(1-p_x)^gamma*log(q_x)
* quality focal loss: QFL(σ) = -|y-σ|^β*((1-y)log(1-σ)+y*log(σ))
* distribution focal loss: DFL(Si,Si+1) = -((yi+1 - y)*log(Si)+(y - yi)*log(Si+1)), Si=(yi+1 - y)/(yi+1 - yi), Si+1=(y - yi)/(yi+1 - yi)