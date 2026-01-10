#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整评估脚本
在目标环境（有GPU）运行完整评估
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import paddle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender import MovieRecommender
from evaluation.enhanced_metrics import EnhancedMetrics
from models.ncf_model import NCF


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="推荐系统评估")

    parser.add_argument("--data_dir", type=str, default=None, help="数据目录路径")
    parser.add_argument("--model_path", type=str, default=None, help="模型文件路径")
    parser.add_argument(
        "--use_features", action="store_true", default=True, help="是否使用特征"
    )
    parser.add_argument(
        "--use_poster", action="store_true", default=False, help="是否使用海报特征"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.json",
        help="输出结果文件",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="批大小")

    return parser.parse_args()


def load_model_and_evaluate(args):
    """加载模型并评估"""
    if args.data_dir is None:
        args.data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("推荐系统 - 完整评估")
    print("=" * 60)

    # 加载推荐系统
    print("\n[1/4] 加载推荐系统...")
    recommender = MovieRecommender(
        data_dir=args.data_dir,
        model_path=args.model_path,
        use_features=args.use_features,
        use_poster=args.use_poster,
    )
    print(f"  用户数: {recommender.n_users}")
    print(f"  电影数: {recommender.n_movies}")

    # 加载测试数据
    print("\n[2/4] 加载测试数据...")
    test_data = pd.read_csv(
        os.path.join(args.data_dir, "data", "processed", "test_ratings.csv")
    )
    print(f"  测试样本数: {len(test_data)}")

    # 评估
    print("\n[3/4] 评估模型...")

    evaluator = EnhancedMetrics(rating_threshold=3.5)

    # 获取推荐结果
    recommendations = {}
    ground_truth = {}

    test_users = test_data["user_id"].unique()[:100]  # 只评估前100个用户

    for user_id in test_users:
        user_test_data = test_data[test_data["user_id"] == user_id]
        recs = recommender.recommend_personalized(user_id, n=10, method="hybrid")
        recommendations[user_id] = recs
        ground_truth[user_id] = user_test_data["movie_id"].tolist()

    # 获取所有电影
    all_movies = recommender.all_movies

    # 计算预测 (使用简单基线)
    y_true = test_data["rating"].values
    y_pred = []

    global_avg = test_data["rating"].mean()

    for _, row in test_data.iterrows():
        user_id = row["user_id"]
        movie_id = row["movie_id"]

        user_ratings = recommender.ratings[recommender.ratings["user_id"] == user_id]
        movie_ratings = recommender.ratings[recommender.ratings["movie_id"] == movie_id]

        user_avg = (
            user_ratings["rating"].mean() if len(user_ratings) > 0 else global_avg
        )
        movie_avg = (
            movie_ratings["rating"].mean() if len(movie_ratings) > 0 else global_avg
        )

        pred = (user_avg + movie_avg + global_avg) / 3
        y_pred.append(pred)

    y_pred = np.array(y_pred)

    # 完整评估
    results = evaluator.full_evaluation(
        y_true=y_true,
        y_pred=y_pred,
        recommendations=recommendations,
        ground_truth=ground_truth,
        all_items=all_movies,
        similarity_matrix=recommender.movie_similarity_matrix,
        item_popularity=recommender.movie_popularity,
        total_interactions=len(recommender.ratings),
    )

    # 打印报告
    evaluator.print_report(results)

    # 保存结果
    print("\n[4/4] 保存评估结果...")

    # 处理numpy类型以便JSON序列化
    serializable_results = {}
    for key, value in results.items():
        if key == "ConfusionMatrix":
            serializable_results[key] = value.tolist()
        elif isinstance(value, (np.floating, float)):
            serializable_results[key] = float(value)
        elif isinstance(value, (np.integer, int)):
            serializable_results[key] = int(value)
        elif isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = str(value)

    output_path = os.path.join(args.data_dir, args.output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)

    print(f"  评估结果已保存到: {output_path}")

    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)

    return results


if __name__ == "__main__":
    args = parse_args()
    results = load_model_and_evaluate(args)
