# 使用PaddlePaddle官方GPU镜像
FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.2.2-gpu-cuda13.0-cudnn9.13

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . .

# 安装项目依赖（除paddlepaddle-gpu外）
RUN pip install --no-cache-dir \
    pandas>=1.0.0 \
    numpy>=1.19.0 \
    wget>=3.2 \
    Pillow>=8.0.0 \
    tqdm>=4.50.0 \
    scikit-learn>=0.24.0 \
    matplotlib>=3.3.0 \
    seaborn>=0.11.0

# 暴露端口
EXPOSE 8888

# 设置默认命令
CMD ["python", "main.py"]
