# 电影推荐系统

基于PaddlePaddle的个性化电影推荐系统，支持NCF和SASRec两种推荐模型，实现热门推荐、新品推荐、个性化推荐、序列推荐等多种推荐路径。

## 目录

- [快速开始](#快速开始)
- [本地开发](#本地开发)
- [项目结构](#项目结构)
- [配置参数](#配置参数)
- [评估指标](#评估指标)

## 快速开始

### Docker部署（推荐）

```bash
# 1. 构建并运行容器
docker compose up --build

# 2. 或进入容器交互式使用
docker run --gpus all --name paddle -it -v $PWD:/app \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.2.2-gpu-cuda13.0-cudnn9.13 /bin/bash

# 在容器内运行
python main.py              # 初始化
python train.py --epochs=20 # 训练NCF
python train_sasrec.py      # 训练SASRec
python scripts/quick_test.py # 测试
```

### 本地开发

```bash
# CPU版本（本地开发）
pip install -r requirements-cpu.txt

# GPU版本（需要CUDA环境）
python -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/
pip install -r requirements-gpu.txt
```

### 2. 数据准备

```bash
# 下载数据并预处理
python main.py
```

这将自动：
- 下载ml-1m数据集
- 下载电影海报数据
- 处理数据并提取特征
- 计算相似度矩阵

### 3. 模型训练

```bash
# 训练NCF模型
python train.py --epochs 10 --batch_size 256 --use_poster

# 训练SASRec序列推荐模型
python train_sasrec.py --epochs 20 --batch_size 64 --max_len 50
```

### 4. 生成推荐

```bash
# 测试推荐系统（包含NCF和SASRec）
python recommender.py

# 快速测试（推荐方法对比）
python scripts/quick_test.py
```

## 项目结构

```
paddle_movie_recommender/
├── data/                    # 数据相关
│   ├── download_and_process.py  # 数据下载和预处理
│   ├── dataset.py           # 数据集定义
│   └── sequence_dataset.py  # SASRec序列数据处理
├── models/                  # 模型相关
│   ├── ncf_model.py         # NCF模型定义
│   ├── sasrec_model.py      # SASRec序列推荐模型
│   └── poster_feature.py    # 海报特征提取
├── evaluation/              # 评估相关
│   └── evaluator.py         # 评估指标计算
├── scripts/                 # 脚本相关
│   └── quick_test.py        # 快速测试脚本
├── recommender.py           # 推荐系统主程序
├── train.py                 # NCF训练脚本
├── train_sasrec.py          # SASRec训练脚本
├── main.py                  # 入口脚本
├── PROJECT_SUMMARY.md       # 项目完成总结
└── README.md               # 说明文档
```

## 配置参数

### train.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --data_dir | 当前目录 | 数据目录 |
| --batch_size | 256 | 批次大小 |
| --epochs | 10 | 训练轮数 |
| --learning_rate | 0.001 | 学习率 |
| --use_features | True | 使用特征 |
| --use_poster | False | 使用海报特征 |
| --save_dir | models | 模型保存目录 |

### train_sasrec.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --data_dir | 当前目录 | 数据目录 |
| --batch_size | 64 | 批次大小 |
| --epochs | 20 | 训练轮数 |
| --learning_rate | 0.001 | 学习率 |
| --max_len | 50 | 序列最大长度 |
| --hidden_units | 64 | 隐藏单元数 |
| --num_heads | 2 | 注意力头数 |
| --num_blocks | 2 | Transformer块数 |
| --dropout_rate | 0.5 | Dropout比率 |

## 评估指标

系统支持以下评估指标：

- **MAE**：平均绝对误差
- **RMSE**：均方根误差
- **Accuracy**：预测精度
- **Precision@K**：精确率
- **Recall@K**：召回率
- **F1@K**：F1分数
- **NDCG@K**：归一化折损累积增益
- **HitRate@K**：命中率
- **Confusion Matrix**：混淆矩阵
- **Coverage**：覆盖率
- **Diversity**：多样性

## 使用示例

```python
from recommender import MovieRecommender

# 初始化推荐系统（支持NCF和SASRec）
recommender = MovieRecommender(
    data_dir='./data',
    model_path='./models/ncf_model.pdparams',
    sasrec_model_path='./models/sasrec_model.pdparams',
    use_features=True,
    use_poster=True
)

# 综合推荐（混合NCF、SASRec、相似用户、相似电影）
recommendations = recommender.recommend(user_id=1, n=10, method='hybrid')

# NCF模型推荐
ncf_recs = recommender.recommend_personalized(user_id=1, n=5, method='model')

# SASRec序列推荐
sasrec_recs = recommender.recommend_personalized(user_id=1, n=5, method='sasrec')

# 相似用户推荐
similar_user_recs = recommender.recommend_personalized(user_id=1, n=5, method='user_sim')

# 相似电影推荐
similar_movie_recs = recommender.recommend_personalized(user_id=1, n=5, method='movie_sim')

# 新用户推荐
new_user_recs = recommender.recommend('new_user', n=10)
```

## 推荐方法

| 方法 | 说明 | 适用场景 |
|------|------|----------|
| model | NCF深度学习推荐 | 融合用户和电影特征的深度推荐 |
| sasrec | SASRec序列推荐 | 基于用户行为序列的时序推荐 |
| user_sim | 相似用户推荐 | 基于用户画像的协同过滤 |
| movie_sim | 相似电影推荐 | 基于内容相似的协同过滤 |
| hybrid | 混合推荐 | 综合多种方法的最优推荐 |

## Git管理

训练生成的文件已加入 `.gitignore`，包括：
- `*.pdparams` - 模型参数文件
- `data/processed/*.pkl` - 处理后的数据文件
- `data/processed/*.txt` - 序列数据文件
- `__pycache__/` - Python缓存文件
- `*.log` - 日志文件

请勿将这些文件提交到git仓库。

## 海报特征影响验证

系统支持验证电影海报特征对推荐结果的影响：

```bash
# 不使用海报特征训练
python train.py --use_poster False

# 使用海报特征训练
python train.py --use_poster True

# 比较两者的评估指标
```

## 依赖列表

```
paddlepaddle>=2.0.0
pandas>=1.0.0
numpy>=1.19.0
wget>=3.2
Pillow>=8.0.0
tqdm>=4.50.0
scikit-learn>=0.24.0
```

## 数据集

本项目使用[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)数据集，包含：
- 6,040 用户
- 3,952 电影
- 1,000,209 评分

每个用户至少评分20部电影。

## 模型说明

### NCF (Neural Collaborative Filtering)

NCF模型融合了广义矩阵分解（GMF）和多层感知机（MLP），并支持用户特征、电影类型特征和海报视觉特征的融合。

### SASRec (Self-Attentive Sequential Recommendation)

SASRec基于Transformer架构，使用自注意力机制捕获用户行为序列中的时序依赖关系，适合推荐用户可能感兴趣的下一个物品。

## 参考资料

1. He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). Neural collaborative filtering. WWW.
2. Kang, W., & McAuley, J. (2018). Self-attentive sequential recommendation. ICDM.
3. ResNet: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.
4. MovieLens Dataset: https://grouplens.org/datasets/movielens/1m/
