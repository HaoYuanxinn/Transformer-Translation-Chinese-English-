FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# 安装基础工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda，用来固定 Python 3.10
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_24.3.0-0-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# 固定 Python 版本
RUN conda install -y python=3.10 pip && \
    conda clean -afy

# 先复制依赖文件，安装依赖
COPY requirements.txt /workspace/requirements.txt

RUN python -m pip install --upgrade pip && \
    pip install -r /workspace/requirements.txt

# 再复制整个项目
COPY . /workspace

# 默认进入 bash
CMD ["bash"]