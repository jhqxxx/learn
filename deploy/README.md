### 部署平台
#### onnx
* onnxruntime
* opencv-dnn
    - 使用opencv-dnn部署onnx模型
    - 很久之前试过，用cuda时有bug
    - cv::mat的minMaxLoc方法,可以获取mat中最大和最小值及他们的索引
* libtorch
    - pytorch的C++ api
    - 加载使用torch.jit.trace转换的torch.jit.ScriptModule模型
    - 使用torch::jit::script::Module定义模型
    - 使用torch::jit::load加载torchscript模型
* TorchScipt
    - 是pytorch的一种模型部署方式