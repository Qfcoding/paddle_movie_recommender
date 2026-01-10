# 电影推荐系统

基于PaddlePaddle的个性化电影推荐系统，实现热门推荐、新品推荐、个性化推荐三种推荐路径。

## 功能特性

### 推荐路径
- **热门推荐**：基于评分频率选择最热门的电影
- **新品推荐**：基于首映时间推荐最新上映的电影
- **个性化推荐**：基于NCF模型和用户/电影相似度进行个性化推荐

### 核心特性
- 支持用户相似度推荐和电影相似度推荐
- 支持新用户冷启动推荐
- 集成电影海报特征（Poster）增强推荐效果
- 完善的评估指标体系（MAE, RMSE, Precision, Recall, NDCG, Coverage, Diversity）

## 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
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

# 使用海报特征训练
python train.py --use_poster
```

### 4. 生成推荐

```bash
# 测试推荐系统
python recommender.py
```

## 项目结构

```
paddle_movie_recommender/
├── data/                    # 数据相关
│   ├── download_and_process.py  # 数据下载和预处理
│   └── dataset.py           # 数据集定义
├── models/                  # 模型相关
│   ├── ncf_model.py         # NCF模型定义
│   └── poster_feature.py    # 海报特征提取
├── evaluation/              # 评估相关
│   └── evaluator.py         # 评估指标计算
├── recommender.py           # 推荐系统主程序
├── train.py                 # 训练脚本
├── main.py                  # 入口脚本
├── requirements.txt         # 依赖列表
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

# 初始化推荐系统
recommender = MovieRecommender(
    data_dir='./data',
    model_path='./models/ncf_model.pdparams',
    use_features=True,
    use_poster=True
)

# 综合推荐（混合热门、新品、个性化）
recommendations = recommender.recommend(user_id=1, n=10, method='hybrid')

# 个性化推荐
personalized = recommender.recommend_personalized(user_id=1, n=5, method='model')

# 相似用户推荐
similar_user_recs = recommender.recommend_personalized(user_id=1, n=5, method='user_sim')

# 相似电影推荐
similar_movie_recs = recommender.recommend_personalized(user_id=1, n=5, method='movie_sim')

# 新用户推荐
new_user_recs = recommender.recommend('new_user', n=10)
```

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

## 参考资料

1. He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). Neural collaborative filtering. WWW.
2. ResNet: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.
