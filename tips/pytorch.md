torch.nn与torch.nn.function
    大部分torch.nn中的层class都有nn.function对应，其区别是：
        * nn.Module实现的layer是由class Layer(nn.Module)定义的特殊类，会自动提取可学习的参数nn.Parameter
        * nn.functional中的函数更像是纯函数，由def function(input)定义
    由于两者性能差异不大，所以具体使用取决于具体情况和个人喜欢
        * 对于激活函数和池化层，没有可学习的参数，一般使用nn.functional
        * 其他有学习参数的部分则使用nn.Module
        * 由于Dropout在训练和测试时操作不同，建议使用nn.Module,其可以通过model.eval加以区分