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


## 项目结构

transformer/
│
├── `data/`                  # 数据文件
│   └── `json/`
│
├── `tokenizer/`             # SentencePiece 分词模型
│
├── `results/`               # 训练结果与曲线
│
├── `transformer_model.py`   # 模型结构定义
├── `beam_decoder.py`        # Beam Search 解码
├── `train.py`               # 训练入口
├── `translate.py`           # 推理入口
├── `data_loader.py`         # 数据加载与 mask 构造
├── `utils.py`               # 工具函数
├── `config.py `             # 参数配置
└── `requirements.txt`

## 环境

- python=3.10
- torch==1.12.1  CUDA=11.3
- tqdm==4.67.1
- sentencepiece==0.2.1
- sacrebleu==2.5.1
- numpy==1.23.3
- Matplotlib==3.10.8

## 训练方法
- 运行 train.py 开始训练，模型权重保存位置 results/best_bleu_xx.pth
- 运行 translate.py 推理



