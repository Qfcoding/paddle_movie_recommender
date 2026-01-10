#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主入口脚本
整合数据处理、特征提取、模型训练和推荐生成
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.download_and_process import main as process_data
from models.poster_feature import PosterFeatureExtractor, build_poster_database
import pandas as pd
import paddle


def check_gpu():
    """检查是否有GPU支持"""
    try:
        return paddle.is_compiled_with_cuda()
    except Exception:
        return False


def setup_project():
    """项目初始化"""
    print("=" * 60)
    print("电影推荐系统 - 项目初始化")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    poster_dir = os.path.join(data_dir, "posters")
    output_dir = os.path.join(data_dir, "processed")

    has_gpu = check_gpu()
    print(f"\n硬件检测: GPU = {'有' if has_gpu else '无 (本地模式)'}")

    # 确保data目录存在
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 1. 下载和预处理数据
    print("\n[1/4] 下载和预处理数据...")
    import subprocess

    result = subprocess.run(
        [sys.executable, os.path.join(data_dir, "download_and_process.py")],
        cwd=data_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"数据处理失败: {result.stderr}")
    else:
        print(result.stdout)

    # 2. 构建海报数据库
    print("\n[2/4] 构建海报数据库...")
    if os.path.exists(poster_dir) and len(os.listdir(poster_dir)) > 0:
        movies_df = pd.read_csv(os.path.join(data_dir, "processed", "movies.csv"))
        movie_id_to_path = build_poster_database(poster_dir, output_dir, movies_df)
        print(f"找到 {len(movie_id_to_path)} 张海报")
    else:
        print("海报目录为空或不存在，跳过海报处理")

    # 3. 提取海报特征
    print("\n[3/4] 提取海报特征...")
    poster_features_file = os.path.join(output_dir, "poster_features.pkl")
    if os.path.exists(poster_dir) and len(os.listdir(poster_dir)) > 0:
        if os.path.exists(poster_features_file):
            print("海报特征已存在，跳过提取")
        elif has_gpu:
            extractor = PosterFeatureExtractor(poster_dir, output_dir, use_gpu=True)
            movies_df = pd.read_csv(os.path.join(data_dir, "processed", "movies.csv"))
            movie_ids = movies_df["movie_id"].tolist()

            mapping_file = os.path.join(output_dir, "poster_mapping.pkl")
            if os.path.exists(mapping_file):
                import pickle

                with open(mapping_file, "rb") as f:
                    movie_id_to_path = pickle.load(f)
                extractor.extract_poster_features(movie_ids, movie_id_to_path)
            else:
                print("海报映射文件不存在")
        else:
            print("本地无GPU，跳过海报特征提取")
            print("提示: 请在有GPU的环境重新运行以提取海报特征")
    else:
        print("海报目录不存在，跳过海报特征提取")

    # 4. 计算相似度矩阵
    print("\n[4/4] 初始化推荐系统...")
    from recommender import MovieRecommender

    recommender = MovieRecommender(data_dir=data_dir)

    print("计算用户相似度...")
    recommender.compute_user_similarity(method="features")

    print("计算电影相似度...")
    recommender.compute_movie_similarity(method="features")

    print("\n" + "=" * 60)
    print("项目初始化完成！")
    print("=" * 60)
    print("\n下一步:")
    print("  1. 运行 train.py 训练模型")
    print("  2. 运行 recommender.py 测试推荐")
    print("  3. 打开 docs/main.ipynb 查看完整文档")


if __name__ == "__main__":
    setup_project()
