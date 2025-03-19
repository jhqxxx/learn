<!--
 * @Author: jhq
 * @Date: 2025-02-08 14:15:50
 * @LastEditTime: 2025-03-14 16:03:49
 * @Description: 
-->
##### 张量操作
* torch.empty(*sizes, out=None, dtype=None, layout=torch.strided,  device=None, requires_grad=False, pin_memory=False) → TensorP:创建一个未初始化的张量，张量形状由sizes指定。
* torch.nn.init.uniform_(tensor, a=0, b=1, *, generator=None) → None:将张量tensor的元素初始化为在[a, b]范围内均匀分布的随机数。
* torch.nn.Parameter(data, requires_grad=True) → Parameter:将不可训练的tensor转换成可以训练的tensor，创建一个参数对象，参数对象是一个特殊的张量，具有requires_grad属性，并且可以自动求导。
* torch.clamp(input, min, max, *, out=None) -> Tensor:将输入张量中的元素大小限制在min和max之间，并返回结果张量。
* torch.log(input) -> Tensor:以自然数e为底的对数函数。
* ？？torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None) → Module:二维转置卷积层。
    - stride为1时，按正常卷积计算输出尺寸
    - stride>1时：
        - 需要插值，在输入数据的每两个相邻值之间插入(stride-1)个0，输入为H,则有H-1个间隙，则一共插入(stride-1)*(H-1)个0
        - 则新的特征图尺寸为 H + (stride-1)*(H-1)
        - new_padding = kernel_size - padding -1:？？？
        - new_stride = 1
        - new_kernel_siez = kernel_size        
        - 输出特征图计算：(H + (stride-1)*(H-1) + 2*(kernel_size - padding - 1))/1 + 1 + output_padding
        - 化简为：(H-1)*stride + kernel_size - 2*padding + output_padding
* ？？torch.nn.Upsample(size=None, scale_factor=None, mode='nearest', align_corners=None) → Module:对输入张量进行上采样。
* torch.nn.functional.pad(input, pad, mode='constant', value=0):对输入张量进行填充。
* torch.chunk(input, chunks, dim=0) -> Tuple[Tensor]:将输入张量按照dim维度切分为chunks个子张量，并返回一个元组。
* torch.einsum():爱因斯坦求和
* torch.unsqueeze(input, dim) -> Tensor:在指定维度上增加一个维度，并返回结果张量。
* torch.nn.functional.normalize(input, p=2, dim=None, eps=1e-12, out=None) -> Tensor:对输入张量进行归一化。p:范数类型，默认为2，L2范数。
* torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False, out=None) -> Tensor:对输入张量进行插值。
* tensor.split(split_size_or_sections, dim=0) -> List[Tensor]:将输入张量按照dim维度切分为split_size_or_sections中指定大小的子张量，并返回一个列表。
* 索引
    - a = torch.rand(4, 3, 28, 28) # a.shape = (b,c,h,w)
    - a[..., :2]: 所有b,c,h维度，取01列
    - a[..., 0:28:2]: 等同于a[..., ::2]，从0开始，步长为2
    - a[..., 1:28:2]: 等同于a[..., 1::2]，从1开始，步长为2
    - 上述操作常用于取偶数列或基数列

* register_buffer(name, tensor, persistent=True)->None:将张量tensor注册为buffer,会自动成为模型中的参数，随着模型移动(gpu/cpu)而移动，但不会随着梯度进行更新