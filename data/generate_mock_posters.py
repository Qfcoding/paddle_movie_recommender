#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成模拟海报特征数据
当真实海报数据不可用时，使用基于电影特征的模拟海报特征
"""

import os
import pickle
import numpy as np
import pandas as pd


def generate_mock_poster_features(data_dir, output_dir):
    """
    生成模拟的海报特征

    基于电影类型和年份生成模拟的海报特征向量
    这样可以验证Poster特征对推荐结果的影响
    """
    print("生成模拟海报特征...")

    # 加载电影数据
    movies_df = pd.read_csv(os.path.join(data_dir, "processed", "movies.csv"))

    # 获取类型列（确保是数值类型）
    genre_cols = [c for c in movies_df.columns if c.startswith("genre_")]

    poster_features = {}

    for _, row in movies_df.iterrows():
        movie_id = row["movie_id"]

        # 基于电影特征生成模拟的海报特征
        # 特征向量包含：
        # 1. 类型one-hot编码的投影
        # 2. 年份编码
        # 3. 一些随机噪声（模拟不同电影间的视觉差异）

        # 类型特征 - 确保是数值类型
        genre_features = np.array(
            [float(row[c]) if pd.notna(row[c]) else 0.0 for c in genre_cols],
            dtype=np.float32,
        )

        # 年份特征 (1维)
        year = row["release_year"] if pd.notna(row["release_year"]) else 1990
        year_norm = (year - 1920) / (2000 - 1920)
        year_feature = np.array([year_norm], dtype=np.float32)

        # 基础特征 (21维)
        base_features = np.concatenate([genre_features, year_feature])

        # 使用电影ID作为随机种子，确保同一电影生成相同的特征
        np.random.seed(int(movie_id))

        # 添加随机噪声使特征更丰富
        noise = np.random.randn(2048 - 21).astype(np.float32) * 0.1

        # 组合特征
        poster_feat = np.concatenate([base_features, noise])

        # 归一化
        poster_feat = poster_feat / (np.linalg.norm(poster_feat) + 1e-8)

        poster_features[movie_id] = poster_feat

    # 保存特征
    output_file = os.path.join(output_dir, "poster_features.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(poster_features, f)

    print(f"生成 {len(poster_features)} 个电影的海报特征")
    print(f"特征维度: {len(list(poster_features.values())[0])}")

    # 同时生成海报映射文件（指向模拟数据）
    movie_id_to_path = {mid: None for mid in poster_features.keys()}
    mapping_file = os.path.join(output_dir, "poster_mapping.pkl")
    with open(mapping_file, "wb") as f:
        pickle.dump(movie_id_to_path, f)

    print(f"海报映射文件已保存: {mapping_file}")
    print(f"注意: 这是模拟的海报特征，用于验证Poster特征对推荐结果的影响")
    print(f"真实场景中应使用实际的海报图片提取特征")

    return poster_features


def main():
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(data_dir, "processed")

    generate_mock_poster_features(data_dir, output_dir)


if __name__ == "__main__":
    main()
