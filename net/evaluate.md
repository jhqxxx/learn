- 精度: 预测正确的样本占预测样本的比例,accuracy = (TP+TN)/(TP+FP+TN+FN)
- 精确率: 预测是正例的结果中，确实是正例的比例，precision = TP/(TP+FP)
- 召回率：所有正例的样本被找出的比例，recall = TP/(TP+FN)
- P-R 曲线：横坐标 recall，纵坐标 precision
- F1-score：(2*precision*recall)/(precision+recall)
- TPR：真正例率，同召回率
- FPR：假正例率，FPR = FP/(FP+TN)
- ROC 曲线：横坐标 FPR，纵坐标 TPR
- AUC：ROC 曲线下的面积
- 敏感性：同召回率
- 特异度：TN/(FP+TN)

- AP: 某一类 P-R 曲线下的面积
- mAP: 所有类别的 AP 值取平均， mAP = (1/n)*sum(precision@i)*rel(i)

- GAN
    - IS
    - FID
    - Mode Score
    - Modified Inception Score
    - AM Score
    - MMD
    - Image Quality Measures
    - SSIM
    - PSNR

* 困惑度 perplexity
* BLEU??
* ROUGE??

