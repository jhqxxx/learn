<!--
 * @Author: jhq
 * @Date: 2025-02-08 14:15:50
 * @LastEditTime: 2025-04-30 11:50:42
 * @Description:
-->

##### 张量操作

- torch.empty(\*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False) → TensorP:创建一个未初始化的张量，张量形状由 sizes 指定。
- torch.nn.init.uniform\_(tensor, a=0, b=1, \*, generator=None) → None:将张量 tensor 的元素初始化为在[a, b]范围内均匀分布的随机数。
- torch.nn.Parameter(data, requires_grad=True) → Parameter:将不可训练的 tensor 转换成可以训练的 tensor，创建一个参数对象，参数对象是一个特殊的张量，具有 requires_grad 属性，并且可以自动求导。
- torch.clamp(input, min, max, \*, out=None) -> Tensor:将输入张量中的元素大小限制在 min 和 max 之间，并返回结果张量。
- torch.log(input) -> Tensor:以自然数 e 为底的对数函数。
- ？？torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None) → Module:二维转置卷积层。
  - stride 为 1 时，按正常卷积计算输出尺寸
  - stride>1 时：
    - 需要插值，在输入数据的每两个相邻值之间插入(stride-1)个 0，输入为 H,则有 H-1 个间隙，则一共插入(stride-1)\*(H-1)个 0
    - 则新的特征图尺寸为 H + (stride-1)\*(H-1)
    - new_padding = kernel_size - padding -1:？？？
    - new_stride = 1
    - new_kernel_siez = kernel_size
    - 输出特征图计算：(H + (stride-1)_(H-1) + 2_(kernel_size - padding - 1))/1 + 1 + output_padding
    - 化简为：(H-1)*stride + kernel_size - 2*padding + output_padding
- ？？torch.nn.Upsample(size=None, scale_factor=None, mode='nearest', align_corners=None) → Module:对输入张量进行上采样。
- torch.nn.functional.pad(input, pad, mode='constant', value=0):对输入张量进行填充。
- torch.chunk(input, chunks, dim=0) -> Tuple[Tensor]:将输入张量按照 dim 维度切分为 chunks 个子张量，并返回一个元组。
- torch.einsum():爱因斯坦求和
- torch.unsqueeze(input, dim) -> Tensor:在指定维度上增加一个维度，并返回结果张量。
- torch.nn.functional.normalize(input, p=2, dim=None, eps=1e-12, out=None) -> Tensor:对输入张量进行归一化。p:范数类型，默认为 2，L2 范数。
- torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False, out=None) -> Tensor:对输入张量进行插值。
- tensor.split(split_size_or_sections, dim=0) -> List[Tensor]:将输入张量按照 dim 维度切分为 split_size_or_sections 中指定大小的子张量，并返回一个列表。
- 索引

  - a = torch.rand(4, 3, 28, 28) # a.shape = (b,c,h,w)
  - a[..., :2]: 所有 b,c,h 维度，取 01 列
  - a[..., 0:28:2]: 等同于 a[..., ::2]，从 0 开始，步长为 2
  - a[..., 1:28:2]: 等同于 a[..., 1::2]，从 1 开始，步长为 2
  - 上述操作常用于取偶数列或基数列

- register_buffer(name, tensor, persistent=True)->None:将张量 tensor 注册为 buffer,会自动成为模型中的参数，随着模型移动(gpu/cpu)而移动，但不会随着梯度进行更新

- libtorch 使用
  - 报错：error while loading shared libraries: libc10.so: cannot open shared object file
  - 解决：libtorch 加环境变量
    - export LIBTORCH_DIR="/home/jhq/depends/libtorch"
    - export LD_LIBRARY_PATH=$LIBTORCH_DIR/lib:$LD_LIBRARY_PATH

- torch.rsqrt() -> 对元素取平方根后再取倒数
  - 1/torch.sqrt()

* nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None)
  - num_embeddings: 词表大小
  - embedding_dim: 词向量维度
  - 将输入的整数索引转换为对应的词向量
  - tokenizer定义了词表大小后，每个单词就转换成立为一个索引
  - 再使用Embedding将索引转换为对应的词向量

* torch.outer(input, vec2):计算两个向量外积-input^T*vec2

* torch.polar(abs, angle): 将绝对值和角度转换为复数
  - out = abs*cos(angle) + abs*sin(angle)*j

* tensor.ndim: tensor.dim()的别名，返回张量维度数量
  - [4, 5, 6] -> 3

* torch.view_as_complex：将实数张量转换为复数张量
* torch.view_as_real(): 将复数张量转换为实数张量

* torch.repeat_interleave(input, repeats, dim=None): 重复输入张量，在指定维度插入重复的元素
* torch.cumsum(input, dim, dtype=None): 对输入张量进行累计操作，返回新张量的每个元素都是原张量中对应位置及之前所有元素的累加和

* torch.multinomial(): 多项式采样，从一个概率分布中，采样n个样本的index
* torch.gather(input, dim, index): 从输入张量中，根据索引获取对应位置的元素
* tensor.detach()
  返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false
* model.fuse(): 将模型中的某些层进行融合，加速模型推理
  * conv,bn
  * conv,bn,relu
  * conv,relu
  * linear,bn
  * bn,relu

* torch.tensor(np.array): 将numpy数组转换为tensor，复制了一份数据,numpy修改，tensor不会修改
* torch.from_numpy(np.array): 将numpy数组转换为tensor，共享数据,numpy修改，tensor也会修改
* tensor.matmul(other): 矩阵乘法
* @: 矩阵乘法