* loss: use candle_nn::{loss};
    * BCE: loss::binary_cross_entropy_with_logit(input, label):该方法内部会对input做sigmoid，然后再计算BCE损失

* candle_core 的某些版本（如 v0.3.x）不支持直接对张量调用 .requires_grad(true)。这个 API 在 PyTorch 中存在，但在 candle 中，梯度追踪是通过计算图自动管理的，并不是所有构建模式都支持显式标记张量为“需要梯度”。
* 使用 Var 类型代替 Tensor: candle-nn 提供了 Var（变量），专门用于表示可训练参数（即需要梯度的张量）。
* candle_nn中的Linear,计算时是input.matmul(weight.t())!!!!