# GPU 配置与云端训练

> 用 CPU 训练适合学习；要做真正的训练，你需要 GPU。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 minutes

## 学习目标

- 使用 `nvidia-smi` 和 PyTorch 的 CUDA API 验证本机 GPU 可用性
- 在 Google Colab 上启用免费的 T4 GPU，用于云端实验
- 基准测试 CPU 与 GPU 的矩阵乘法，并量化加速比
- 用 fp16 的经验法则估算你的显存能容纳多大的模型

## 问题

第 1-3 阶段的大多数课程在 CPU 上就能跑得很好。但当你开始训练 CNN、transformer 或 LLM（第 4 阶段及以后），你就需要 GPU 加速。同一次训练，CPU 上可能要 8 小时，GPU 上可能只要 10 分钟。

你有三种选择：本地 GPU、云端 GPU、或者 Google Colab（免费）。

## 核心概念

```text
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

## 动手搭建

### 选项 1：本地 NVIDIA GPU

先看看你有没有 NVIDIA 显卡：

```bash
nvidia-smi
```

安装带 CUDA 的 PyTorch，并验证：

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 选项 2：Google Colab

1. 打开 [colab.research.google.com](https://colab.research.google.com)
2. Runtime > Change runtime type > T4 GPU
3. 运行 `!nvidia-smi` 验证

你可以把本课程的 notebook 直接上传到 Colab。

### 选项 3：云端 GPU

以 Lambda Labs、RunPod、Vast.ai 为例：

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 没有 GPU？也没问题

大多数课程都支持 CPU。需要 GPU 的课程会明确标注，并通常提供 Colab 的替代路径。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## 动手搭建：GPU vs CPU 基准测试

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## 练习

1. 运行上面的基准测试，对比 CPU 与 GPU 耗时
2. 如果你没有 GPU，就在 Google Colab 上跑，并对比结果
3. 看看你的显存有多少，并估算能放下多大的模型（经验法则：fp16 每个参数约 2 字节）

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| CUDA | "GPU programming" | NVIDIA 的并行计算平台，让你把代码跑在 GPU 上 |
| VRAM | "GPU memory" | GPU 上的显存，独立于系统内存；它限制了模型大小 |
| fp16 | "Half precision" | 16-bit 浮点数，内存占用约为 fp32 的一半且精度损失很小 |
| Tensor Core | "Fast matrix hardware" | 专门做矩阵乘法的 GPU 单元，通常比普通核心快 4-8 倍 |

