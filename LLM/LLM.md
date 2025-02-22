<!--
 * @Author: jhq
 * @Date: 2025-02-08 14:16:10
 * @LastEditTime: 2025-02-22 20:26:48
 * @Description:
-->

###### 量化

- 参考<https://huggingface.co/blog/zh/hf-bitsandbytes-integration>
- 参考<https://www.cnblogs.com/huggingface/p/17816374.html>
- 模型大小由其参数量及其精度决定，精度通常为 float32、float16、bfloat16 之一
- 理想情况下训练和推理都应该在 FP32 中完成，但 FP32 比 FP16/BF16 慢两倍,因此实践中常常使用混合精度方法，其中使用 FP32 权重作为精确的主权重，
  而使用 FP16/BF16 权重进行前向和后向传播计算以提高训练速度，最后在梯度更新阶段再使用 FP16/BF16 梯度更新 FP32 主权重。
- 使用低精度，推理结果质量也下降了
- 引入 8 位量化-1 字节，量化过程是从一种数据类型舍入到另一种数据类型，量化是一个有噪过程，会导致信息丢失，是一种有损压缩
  - 零点量化
  - 最大绝对值量化
  - 他们都将浮点值映射为更紧凑的 Int8 值
- LLM.int8(): 性能下降是由离群特征引起的，使用自定义阈值提取离群值，并将矩阵分解为两部分，离群部分用 FP16 表示
- FP8
- FP4
- QLoRA: 使用 4bit 量化来压缩预训练模型，冻结基础模型参数，并将相对少量的可训练参数以低秩适配器的形式添加到模型中

###### 模型微调

- PEFT:parameter efficient fine-tuning 参数高效微调
- QLoRA：低秩适应

###### 模型框架

- Retrieval Augmented Generation-RAG:检索增强生成,外部知识库数据抽取，喂给模型
- Mixture of Experts-MoE:混合专家模型
- Switchable Sparse Dense Learning-SSD:可切换稀疏稠密学习
- Long Sequence:长文本
  - sparse attention:稀疏注意力
    1. sliding window attention:滑动窗口
    2. global attention:全局注意力
    3. context-based sparse attention:基于上下文的稀疏注意力
  - Memory-based methods
  - Linear Attention:线性注意力
  - State Space model:状态空间模型
    1. Mamba
- Scaling law:扩展规律

幻觉问题：错误回答

RLHF

多模态
ViT+LLM
LLM+Diffusion model
open sora

Autonomous agent
controller
preceiver
tool set
environment

XAgent
Inner-loop
Multi-Agent
chat chain

MiniCPM

vllm 安装

台湾大学李宏毅生成式人工智能课程链接：
https://speech.ee.ntu.edu.tw/~hylee/genai/2024-spring.php

1.  作业三问题记录： 1. bitsandbytes 包找不到 cuda 1. 报错如下：RuntimeError:
    CUDA Setup failed despite GPU being available. Please run the following command to get more information:

            python -m bitsandbytes

            Inspect the output of the command and see if you can locate CUDA libraries. You might need to add them
            to your LD_LIBRARY_PATH. If you suspect a bug,

            2. 已有环境：win11-cuda12.6-torch2.6.0+cu126
            3. 参考：<https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/docs/source/installation.mdx>在自己电脑上编译安装
            4. cmake -DCOMPUTE_BACKEND=cuda -S . 遇到问题：
                - Selecting Windows SDK version 10.0.19041.0 to target Windows 10.0.19044.
                -- The CXX compiler identification is MSVC 19.29.30139.0
                CMake Error at C:/Program Files/CMake/share/cmake-3.24/Modules/CMakeDetermineCompilerId.cmake:491 (message):
                No CUDA toolset found.
                - 参考：https://github.com/NVlabs/tiny-cuda-nn/issues/164
                - 但是我的vs安装版本有好几个，路径也乱七八糟，找了几个路径结果下面都有cuda12.6相关的东西，另外又找到一个装在D盘的vs下面没有(我也不知道vs为什么会在D盘,不知道当时装的时候咋想的)，复制过去之后上面的问题解决了，但是有新的问题：
                - Compiling the CUDA compiler identification source file
                    “CMakeCUDACompilerId.cu” failed.
                - 参考：https://blog.csdn.net/qq_26157437/article/details/129834852
                - 但是我修改的时候给我说没有权限保存，记事本用管理员身份打开再编辑，就可以保存了
                - cmake成功，输出如下：
                - bitsandbytes>cmake -DCOMPUTE_BACKEND=cuda -S .

    -- Selecting Windows SDK version 10.0.22000.0 to target Windows 10.0.22631.
    -- Configuring bitsandbytes (Backend: cuda)
    -- The CUDA compiler identification is NVIDIA 12.6.77
    -- Detecting CUDA compiler ABI info
    -- Detecting CUDA compiler ABI info - done
    -- Check for working CUDA compiler: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe - skipped
    -- Detecting CUDA compile features
    -- Detecting CUDA compile features - done
    -- Found CUDAToolkit: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include (found version "12.6.77")
    -- CUDA Version: 126 (12.6.77)
    -- CUDA Compiler: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe
    -- CUDA Capabilities Available: 50;52;53;60;61;62;70;72;75;80;86;87;89;90
    -- CUDA Capabilities Selected: 50;52;53;60;61;62;70;72;75;80;86;87;89;90
    -- CUDA Targets: 50-real;52-real;53-real;60-real;61-real;62-real;70-real;72-real;75-real;80-real;86-real;87-real;89-real;90
    -- CUDA NVCC Flags: -D_WINDOWS -Xcompiler=" /EHsc" --use_fast_math
    -- Configuring done (7.9s)
    -- Generating done (0.0s)

            5. 按步骤3成功安装bitsandbytes,再次测试成功未报错
        2. 模型下载："MediaTek-Research/Breeze-7B-Instruct-v0_1"，很慢
            1. 使用hf-mirror镜像，找到一个仓库：<https://github.com/LetheSec/HuggingFace-Download-Accelerator>
            2. 仓库的文件下载倒是很快，但是老是断，报错：
                * xception: Error while removing corrupted file: 另一个程序正在使用此文件，进程无法访问。 (os error 32)
                * 参考：<https://github.com/LetheSec/HuggingFace-Download-Accelerator/issues/33>
                * 修改之后变慢很多，后面直接卡住不动了
                * 没有用这个脚本
                * 尝试浏览器下载，老是断，不太可行
            3. 使用国内镜像下载，参考<https://hf-mirror.com/>中方法2，有断点续传，停了接着上次的命令下载
            4. 下载后transformers加载模型直接指定模型所在路径、

2.  作业四问题记录：
