<!--
 * @Descripttion: 
 * @version: 
 * @Author: jhq
 * @Date: 2022-09-20 23:32:49
 * @LastEditors: jhq
 * @LastEditTime: 2022-09-20 23:34:19
-->
### 卷积
* 普通卷积：
    - 特征图Hin*Win*Din * 滤波器h*w*Din*Dout -> 输出Hout*Wout*Dout
    - 特征图大小i, kernei size=k, padding=p, stride=s,输出为：lower_bound((1+2p-k)/s)+1
    
* 1*1卷积：H*W*D  *  1*1*D  输出为H*W*1， 如果执行N次1*1卷积，并将结果连接在一起，得到 H*W*N的输出  
    * 降低维度以实现高效计算
    * 高效的低维嵌入，或特征池
    * 卷积后再次应用非线性
* 分组卷积：
    - 3*3的kernel可分为 3*1 和 1*3 两个kernel，输入特征图首先和3*1的kernel卷积，然后再和1*3的kernel卷积
    - 减少参数
    - 减少矩阵乘法
* DW卷积/离散卷积：
    - 通过在卷积核元素之间插入空格来“扩张”卷积核，扩充参数取决于我们想如何扩大卷积核
    - 可扩大输入的感受野，而不增加kernel的尺寸
* PW卷积
* 空洞卷积
* 转置卷积：  
    逆向的卷积，要进行上采样
    输入n*n, 填充p*p的0边缘，3*3的卷积核，stride=1, 输出为(n+2*p-k)/s+1