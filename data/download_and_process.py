#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据下载和预处理脚本
下载ml-1m数据集并处理为推荐系统可用的格式
"""

import os
import wget
import zipfile
import pandas as pd
import re
from datetime import datetime

# 配置
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
POSTER_URL_BASE = "http://files.grouplens.org/datasets/movielens/ml-1m-posters.zip"


def download_ml1m():
    """下载ml-1m数据集"""
    print("正在下载ml-1m数据集...")
    ml1m_zip = os.path.join(DATA_DIR, "ml-1m.zip")

    if os.path.exists(os.path.join(DATA_DIR, "ml-1m", "ratings.dat")):
        print("ml-1m数据集已存在，跳过下载")
        return True

    try:
        wget.download(ML1M_URL, out=ml1m_zip)
        print(f"\n下载完成: {ml1m_zip}")

        # 解压
        print("正在解压...")
        with zipfile.ZipFile(ml1m_zip, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("解压完成")

        # 删除zip文件
        os.remove(ml1m_zip)
        return True

    except Exception as e:
        print(f"下载失败: {e}")
        return False


def download_posters():
    """下载电影海报数据集"""
    print("正在下载电影海报数据集...")
    poster_zip = os.path.join(DATA_DIR, "ml-1m-posters.zip")
    poster_dir = os.path.join(DATA_DIR, "posters")

    if os.path.exists(poster_dir) and len(os.listdir(poster_dir)) > 1500:
        print("海报数据集已存在且完整，跳过下载")
        return True

    try:
        wget.download(POSTER_URL_BASE, out=poster_zip)
        print(f"\n下载完成: {poster_zip}")

        # 解压
        print("正在解压...")
        with zipfile.ZipFile(poster_zip, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("解压完成")

        # 移动到正确位置
        extracted_dir = os.path.join(DATA_DIR, "ml-1m-posters")
        if os.path.exists(extracted_dir):
            import shutil

            for f in os.listdir(extracted_dir):
                src = os.path.join(extracted_dir, f)
                dst = os.path.join(poster_dir, f)
                if os.path.isfile(src):
                    shutil.move(src, dst)
            shutil.rmtree(extracted_dir)

        # 删除zip文件
        os.remove(poster_zip)
        return True

    except Exception as e:
        print(f"海报下载失败: {e}")
        print("将跳过海报功能，继续其他功能")
        return False


def extract_release_year(title):
    """从电影标题中提取首映年份"""
    match = re.search(r"\((\d{4})\)$", title)
    if match:
        return int(match.group(1))
    return None


def process_users():
    """处理用户数据"""
    print("处理用户数据...")
    users_file = os.path.join(DATA_DIR, "ml-1m", "users.dat")

    # 原始格式: UserID::Gender::Age::Occupation::Zip-code
    columns = ["user_id", "gender", "age", "occupation", "zipcode"]
    users_df = pd.read_csv(
        users_file,
        sep="::",
        header=None,
        names=columns,
        encoding="latin-1",
        engine="python",
    )

    # 性别编码
    users_df["gender_encoded"] = (users_df["gender"] == "M").astype(int)

    # 保留原始邮编（作为字符串）
    users_df["zipcode"] = users_df["zipcode"].astype(str)

    # 保存处理后的数据
    output_file = os.path.join(DATA_DIR, "processed", "users.csv")
    users_df.to_csv(output_file, index=False)
    print(f"用户数据处理完成: {len(users_df)} 用户")

    return users_df


def process_movies():
    """处理电影数据"""
    print("处理电影数据...")
    movies_file = os.path.join(DATA_DIR, "ml-1m", "movies.dat")

    # 原始格式: MovieID::Title::Genres
    columns = ["movie_id", "title", "genres"]
    movies_df = pd.read_csv(
        movies_file,
        sep="::",
        header=None,
        names=columns,
        encoding="latin-1",
        engine="python",
    )

    # 提取首映年份
    movies_df["release_year"] = movies_df["title"].apply(extract_release_year)

    # 提取纯标题（去掉年份）
    movies_df["clean_title"] = movies_df["title"].apply(
        lambda x: re.sub(r"\s*\(\d{4}\)\s*$", "", x).strip()
    )

    # 类型处理（多标签）
    all_genres = set()
    for genres in movies_df["genres"]:
        all_genres.update(genres.split("|"))
    all_genres = sorted(all_genres)

    # 创建类型one-hot编码
    for genre in all_genres:
        movies_df[f"genre_{genre}"] = movies_df["genres"].apply(
            lambda x: 1 if genre in x.split("|") else 0
        )

    movies_df["genre_list"] = movies_df["genres"].apply(lambda x: x.split("|"))

    # 保存处理后的数据
    output_file = os.path.join(DATA_DIR, "processed", "movies.csv")
    movies_df.to_csv(output_file, index=False)
    print(f"电影数据处理完成: {len(movies_df)} 电影")

    return movies_df


def process_ratings():
    """处理评分数据"""
    print("处理评分数据...")
    ratings_file = os.path.join(DATA_DIR, "ml-1m", "ratings.dat")

    # 原始格式: UserID::MovieID::Rating::Timestamp
    columns = ["user_id", "movie_id", "rating", "timestamp"]
    ratings_df = pd.read_csv(
        ratings_file,
        sep="::",
        header=None,
        names=columns,
        encoding="latin-1",
        engine="python",
    )

    # 转换为可读时间
    ratings_df["datetime"] = ratings_df["timestamp"].apply(
        lambda x: datetime.fromtimestamp(x)
    )
    ratings_df["year"] = ratings_df["datetime"].apply(lambda x: x.year)
    ratings_df["month"] = ratings_df["datetime"].apply(lambda x: x.month)

    # 划分训练集和测试集（按时间，最后一次评分作为测试）
    print("划分训练集和测试集...")
    ratings_df = ratings_df.sort_values(["user_id", "timestamp"])

    train_data = []
    test_data = []

    for user_id, user_ratings in ratings_df.groupby("user_id"):
        if len(user_ratings) >= 2:
            test_data.append(user_ratings.iloc[-1])
            train_data.extend(user_ratings.iloc[:-1].to_dict("records"))
        else:
            train_data.extend(user_ratings.to_dict("records"))

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # 保存数据
    output_dir = os.path.join(DATA_DIR, "processed")
    ratings_df.to_csv(os.path.join(output_dir, "ratings.csv"), index=False)
    train_df.to_csv(os.path.join(output_dir, "train_ratings.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_ratings.csv"), index=False)

    print(f"评分数据处理完成: {len(ratings_df)} 条评分")
    print(f"  - 训练集: {len(train_df)} 条")
    print(f"  - 测试集: {len(test_df)} 条")

    return ratings_df, train_df, test_df


def compute_statistics(users_df, movies_df, ratings_df):
    """计算统计数据"""
    print("计算统计信息...")

    stats = {
        "用户数量": len(users_df),
        "电影数量": len(movies_df),
        "评分数量": len(ratings_df),
        "评分均值": round(ratings_df["rating"].mean(), 3),
        "评分标准差": round(ratings_df["rating"].std(), 3),
        "评分分布": ratings_df["rating"].value_counts().sort_index().to_dict(),
        "年份范围": f"{movies_df['release_year'].min()} - {movies_df['release_year'].max()}",
    }

    # 热门电影（评分数量最多的10部）
    popular_movies = (
        ratings_df.groupby("movie_id").size().sort_values(ascending=False).head(10)
    )
    stats["热门电影ID"] = list(popular_movies.index)

    # 新电影（最近10年上映的）
    recent_year = movies_df["release_year"].max() - 10
    new_movies = movies_df[movies_df["release_year"] >= recent_year].sort_values(
        "release_year", ascending=False
    )
    stats["新电影数量"] = len(new_movies)

    print("统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 保存统计信息
    import json

    stats_file = os.path.join(DATA_DIR, "processed", "statistics.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v
                for k, v in stats.items()
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return stats


def main():
    """主函数"""
    print("=" * 60)
    print("电影推荐系统 - 数据下载与预处理")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)

    # 下载数据集
    if not download_ml1m():
        print("错误: 无法下载ml-1m数据集")
        return False

    # 下载海报数据集（可选）
    download_posters()

    # 处理数据
    users_df = process_users()
    movies_df = process_movies()
    ratings_df, train_df, test_df = process_ratings()

    # 计算统计信息
    stats = compute_statistics(users_df, movies_df, ratings_df)

    print("\n" + "=" * 60)
    print("数据预处理完成！")
    print("=" * 60)

    return True


if __name__ == "__main__":
    main()
