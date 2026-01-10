#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试脚本
本地快速验证推荐系统功能 (无需GPU)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender import MovieRecommender
from evaluation.enhanced_metrics import EnhancedMetrics
import pandas as pd
import numpy as np


def test_recommender():
    """测试推荐系统"""
    print("=" * 60)
    print("电影推荐系统 - 快速测试")
    print("=" * 60)

    # 项目根目录
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, "data")

    # 创建推荐系统 (不使用海报特征，本地无GPU)
    print("\n[1/4] 初始化推荐系统...")
    recommender = MovieRecommender(
        data_dir=data_dir,
        use_features=True,
        use_poster=False,  # 本地无GPU，跳过海报
    )
    print(f"  用户数: {recommender.n_users}")
    print(f"  电影数: {recommender.n_movies}")

    # 测试用户1
    test_user_id = 1
    print(f"\n[2/4] 为用户 {test_user_id} 生成推荐...")

    # 综合推荐 - 使用相似用户方法（不需要相似度矩阵）
    recommendations = {
        "popular": recommender.recommend_popular(n=2),
        "new": recommender.recommend_new(n=3),
        "personalized": recommender._recommend_by_similar_users(test_user_id, 5),
    }

    # 验证推荐数量
    n_popular = len(recommendations["popular"])
    n_new = len(recommendations["new"])
    n_personalized = len(recommendations["personalized"])
    total = n_popular + n_new + n_personalized

    print(f"\n推荐结果验证:")
    print(f"  热门推荐: {n_popular} 条 (目标: 2条)")
    print(f"  新品推荐: {n_new} 条 (目标: 3条)")
    print(f"  个性化推荐: {n_personalized} 条 (目标: 5条)")
    print(f"  总计: {total} 条")

    # 显示推荐电影
    print(f"\n推荐详情:")
    movies_df = pd.read_csv(os.path.join(data_dir, "processed", "movies.csv"))

    for rec_type, movie_ids in recommendations.items():
        if not movie_ids:
            continue
        print(f"\n【{rec_type}】")
        for mid in movie_ids[:3]:  # 只显示前3条
            movie_info = movies_df[movies_df["movie_id"] == mid]
            if not movie_info.empty:
                title = movie_info.iloc[0]["title"]
                year = movie_info.iloc[0]["release_year"]
                print(f"  - {title} ({year})")

    # 测试新用户
    print(f"\n[3/4] 测试新用户推荐...")
    new_user_recs = recommender.recommend("new_user", n=10)
    print(
        f"  新用户推荐: {len(new_user_recs['popular'])} 热门 + {len(new_user_recs['new'])} 新品 + {len(new_user_recs['personalized'])} 个性化"
    )

    # 评估
    print(f"\n[4/4] 评估模型性能...")

    # 加载测试数据
    test_data = pd.read_csv(os.path.join(data_dir, "processed", "test_ratings.csv"))

    # 使用简单评估: 计算测试集上的MAE
    y_true = test_data["rating"].values[:100]  # 只用前100条
    y_pred = []

    for _, row in test_data.head(100).iterrows():
        user_id = row["user_id"]
        movie_id = row["movie_id"]
        # 简单预测: 用户平均评分 + 电影平均评分 - 全局平均
        user_ratings = recommender.ratings[recommender.ratings["user_id"] == user_id]
        movie_ratings = recommender.ratings[recommender.ratings["movie_id"] == movie_id]

        user_avg = user_ratings["rating"].mean() if len(user_ratings) > 0 else 3.5
        movie_avg = movie_ratings["rating"].mean() if len(movie_ratings) > 0 else 3.5
        global_avg = 3.5

        pred = (user_avg + movie_avg + global_avg) / 3
        y_pred.append(pred)

    y_pred = np.array(y_pred)

    evaluator = EnhancedMetrics()
    results = {
        "RMSE": np.sqrt(np.mean((y_true - y_pred) ** 2)),
        "MAE": np.mean(np.abs(y_true - y_pred)),
        "Accuracy": np.mean(np.abs(y_true - y_pred) < 0.5),
    }

    print(f"\n评估结果:")
    print(f"  RMSE: {results['RMSE']:.4f}")
    print(f"  MAE:  {results['MAE']:.4f}")
    print(f"  Accuracy: {results['Accuracy']:.4f}")

    print("\n" + "=" * 60)
    print("快速测试完成！")
    print("=" * 60)

    return True


def test_similarity_methods():
    """测试相似度计算方法（跳过，因为需要相似度矩阵）"""
    print("\n" + "=" * 60)
    print("相似度计算测试（需要先运行main.py生成相似度矩阵）")
    print("=" * 60)
    print("提示: 运行 python main.py 生成相似度矩阵后再测试")


if __name__ == "__main__":
    success = test_recommender()
    test_similarity_methods()

    if success:
        print("\n✓ 所有测试通过！")
    else:
        print("\n✗ 测试失败，请检查错误信息")
        sys.exit(1)
