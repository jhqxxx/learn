<!--
 * @Author: jhq
 * @Date: 2025-02-17 13:13:40
 * @LastEditTime: 2025-02-22 20:30:53
 * @Description: 
-->
##### rknn 部署自己训练的 yolo11n 目标检测模型

###### 在 PC 端

- 环境：对应 python 环境安装 rknn-toolkit2,windows 平台没有这个包，需要用 linux

  1. pip 安装：pip install rknn-toolkit2 -i https://pypi.org/simple
  2. pip 升级：pip install rknn-toolkit2 -i https://pypi.org/simple --upgrade
  3. 本地 wheel 安装
     1. 克隆仓库<https://github.com/airockchip/rknn-toolkit2>
     2. cd 进入 rknn-toolkit2/rknn-toolkit2
     3. 根据不同的 python 版本及处理器架构，选择不同的 requirements 文件和 whl
     4. pip install -r requirements_cpxx.txt
     5. pip install rknn*toolkit2--x.x.x-cpxx-cpxx-manylinux_2_17*<arch>.manylinux2014\_<arch>.whl

- 训练：使用官方步骤即可
- pt 转 onnx: 使用 airockchip 提供的转换脚本<https://github.com/airockchip/ultralytics_yolo11>,

  1. 参考：<https://github.com/airockchip/ultralytics_yolo11/blob/main/RKOPT_README.zh-CN.md>
  2. 注意：
     - 修改 ultralytics/cfg/default.yaml 文件时
       - model:转换的模型路径
       - batch：input batch size，转出的模型的输入维度[batch,3,640,640],改为 1
       - imgsz: 与训练时的 imgsz 一致，没改过默认为 640

- onnx 转 rknn:使用 airockchip 提供的 rknn_model_zoo 中的转换脚本<https://github.com/airockchip/rknn_model_zoo>

  1. 参考：<https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolo11/README.md>中第 4 步

- cppdemo 编译：
  1. 参考：<https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolo11/README.md>中第 7 步
  2. 注意：
     - **修改 coco_80_labels_list.txt 中的标签为自己模型的标签**
     - 将模型相关的测试图片覆盖原来的 bus.jpg
     - 或者新增图片和标签文件，修改 cpp 下的 CMakeLists.txt 中的 install 对应文件部分
     - **修改 cpp/postprocess.h 中宏定义 OBJ_CLASS_NUM 为自己模型的标签数量**
- 将编译后的可执行程序拷贝到目标设备
  scp -r install/rk3588_linux_aarch64/rknn_yolo11_demo/ xxx@192.168.1.xxx:/home/xxx

###### 在目标设备

- 进入 demo 目录下
- export LD_LIBRARY_PATH=./lib
- ./rknn_yolov5_demo model/model_name.rknn model/bus.jpg
- 拷贝 out.jpg 到 PC 端查看结果

###### cpp 直接在目标设备上编译，待测

* rk3588查看npu使用率： cat /sys/kernel/debug/rknpu/load