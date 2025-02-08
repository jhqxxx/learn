<!--
 * @Author: jhq
 * @Date: 2022-11-13 15:19:39
 * @LastEditTime: 2023-02-09 22:32:31
 * @Description: 
-->
torch.nn与torch.nn.function
    大部分torch.nn中的层class都有nn.function对应，其区别是：
        * nn.Module实现的layer是由class Layer(nn.Module)定义的特殊类，会自动提取可学习的参数nn.Parameter
        * nn.functional中的函数更像是纯函数，由def function(input)定义
    由于两者性能差异不大，所以具体使用取决于具体情况和个人喜欢
        * 对于激活函数和池化层，没有可学习的参数，一般使用nn.functional
        * 其他有学习参数的部分则使用nn.Module
        * 由于Dropout在训练和测试时操作不同，建议使用nn.Module,其可以通过model.eval加以区分


model.eval()切换到测试模式，
    * 主要用于通知dropout和batch norm层在train和val模式之间切换
        1. 在train模式下，dropout网络层会按照参数p设置保留激活单元的概率，batchnorm层会继续计算数据的mean和var等参数并更新；
        2. 在val模式下， dropout层会让所有的激活单元都通过，而batchnorm层会停止计算和更新mean和var等参数，直接使用在训练阶段已经学出来的mean和var值；
    * 该模式不会影响各层的gradient计算行为，即gradient计算和存储与training模式一样，只是不进行反向传播；

with torch.no_grad():
    主要用于停止自动求导，以起到加速和节省显存的作用，停止gradient计算，但是并不会影响dropout和batchnorm层的行为。

如果不在意显存大小和计算时间的话，仅使用model.eval()已能够得到正确的validation的结果，而with torch.no_grad()则能加速和节省gpu空间(因为不用计算和存储gradient)

torch.backends.cudnn.deterministic:将flag设置为True的话，每次返回的卷积算法将是确定的，即默认算法，如果配合设置Torch的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的。

torch.backends.cudnn.benchmark = True:将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速，适用场景时网络结构固定

tensor.item()：返回的是tensor中的值，且只能返回单个值(标量)，不能返回向量，适用返回loss等。
tensor.detach()：阻断反向传播，返回值仍为tensor，和原值指向同一块地址
tensor.cpu()：将变量放在cpu上，仍为tensor

torch.cat(tensors, dim=0,*,out=None):将传入的tensor按维度拼接