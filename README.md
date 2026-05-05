# Transformer 中英翻译

本项目基于 PyTorch 从零实现标准 Transformer（Encoder–Decoder）结构，用于中英机器翻译任务。

实现内容包括：

- 多头注意力（Multi-Head Attention）
- 位置编码（Positional Encoding）
- Encoder–Decoder 结构堆叠
- Padding Mask 与 Subsequent Mask 构造
- Teacher Forcing 训练机制
- Beam Search 解码
- BLEU 指标评估
- 训练曲线可视化
- Docker 环境封装与运行

## 项目结构

```text
transformer/
│
├── data/                         # 数据文件
│   └── json/
│
├── tokenizer/                    # SentencePiece 分词模型
│
├── results/                      # 训练结果、模型权重与训练曲线
│
├── transformer_model.py          # 模型结构定义
├── beam_decoder.py               # Beam Search 解码
├── train.py                      # 训练入口
├── translate.py                  # 推理入口
├── data_loader.py                # 数据加载与 mask 构造
├── utils.py                      # 工具函数
├── config.py                     # 参数配置
├── test_env.py                   # 环境测试脚本
│
├── requirements.txt              # Python 依赖列表
├── Dockerfile                    # Docker 镜像构建文件
├── .dockerignore                 # Docker 构建忽略文件
└── README.md                     # 项目说明文档
```

## 环境

- Python 3.10
- PyTorch 1.12.1
- CUDA 11.3
- tqdm 4.67.1
- sentencepiece 0.2.1
- sacrebleu 2.5.1
- numpy 1.23.3
- matplotlib 3.10.8

## 训练方法

运行 `train.py` 开始训练：

```bash
python train.py
```

模型权重默认保存在：

```text
results/best_bleu_xx.pth
```

运行 `translate.py` 进行推理：

```bash
python translate.py
```

## Docker 使用方法

本项目已经提供 `Dockerfile`，可以直接构建包含 CUDA、Python 和项目依赖的 Docker 环境。

### 方式一：下载项目后本地构建 Docker 镜像

克隆项目后，进入项目目录：

```bash
docker build -t transformer-cu113:latest .
```

构建完成后，运行环境测试脚本：

```bash
docker run --rm -it --gpus all transformer-cu113:latest python test_env.py
```

如果输出中出现：

```text
CUDA available: True
```

说明 Docker 容器已经可以正常使用 GPU。

进入容器：

```bash
docker run --rm -it --gpus all transformer-cu113:latest bash
```

进入容器后，默认位于 `/workspace` 目录，可以直接运行训练代码：

```bash
python train.py
```

也可以运行推理代码：

```bash
python translate.py
```

### 方式二：直接下载 Docker Hub 镜像运行

如果不想在本地重新构建镜像，也可以直接从 Docker Hub 下载已经构建好的镜像。

拉取镜像：

```bash
docker pull haoyuanxin/transformer-cu113:latest
```

运行环境测试脚本：

```bash
docker run --rm -it --gpus all haoyuanxin/transformer-cu113:latest python test_env.py
```

进入容器：

```bash
docker run --rm -it --gpus all haoyuanxin/transformer-cu113:latest bash
```

进入容器后运行训练代码：

```bash
python train.py
```

运行推理代码：

```bash
python translate.py
```
