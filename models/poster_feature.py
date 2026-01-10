#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
海报特征提取模块
使用预训练CNN模型提取电影海报的视觉特征
"""

import os
import numpy as np
import paddle
from paddle.vision import transforms
from paddle.vision.models import resnet50
from PIL import Image
import pickle
from tqdm import tqdm


class PosterFeatureExtractor:
    """海报特征提取器"""

    def __init__(self, poster_dir, output_dir, use_gpu=True):
        """
        初始化海报特征提取器

        Args:
            poster_dir: 海报图片目录
            output_dir: 输出特征文件目录
            use_gpu: 是否使用GPU
        """
        self.poster_dir = poster_dir
        self.output_dir = output_dir

        # 检测GPU
        try:
            self.use_gpu = (
                use_gpu
                and hasattr(paddle, "is_compiled_with_cuda")
                and paddle.is_compiled_with_cuda()
            )
        except Exception:
            self.use_gpu = False

        print(f"海报特征提取器初始化: GPU={'是' if self.use_gpu else '否'}")

        # 模型和预处理
        self.model = None
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        os.makedirs(output_dir, exist_ok=True)

    def load_model(self):
        """加载预训练ResNet50模型"""
        print("加载预训练ResNet50模型...")
        model = resnet50(pretrained=True)

        # 移除最后的分类层，获取特征向量
        self.model = paddle.nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
        )

        self.model.eval()
        print("模型加载完成")

    def extract_poster_features(self, movie_ids, movie_id_to_path):
        """
        提取指定电影的海报特征

        Args:
            movie_ids: 电影ID列表
            movie_id_to_path: 电影ID到海报路径的映射

        Returns:
            poster_features: 电影ID到特征的映射
        """
        if self.model is None:
            self.load_model()

        poster_features = {}
        missing_posters = []

        print(f"开始提取 {len(movie_ids)} 张海报的特征...")

        for movie_id in tqdm(movie_ids):
            poster_path = movie_id_to_path.get(movie_id)

            if poster_path is None or not os.path.exists(poster_path):
                missing_posters.append(movie_id)
                continue

            try:
                # 加载图片
                img = Image.open(poster_path).convert("RGB")
                img_tensor = self.transform(img)

                # 添加batch维度
                img_tensor = img_tensor.unsqueeze(0)

                # 提取特征
                with paddle.no_grad():
                    features = self.model(img_tensor)
                    features = features.reshape([features.shape[0], -1])
                    features = features.numpy()[0]  # 获取特征向量

                poster_features[movie_id] = features

            except Exception as e:
                print(f"处理电影 {movie_id} 的海报时出错: {e}")
                missing_posters.append(movie_id)

        print(f"\n特征提取完成:")
        print(f"  - 成功提取: {len(poster_features)} 张")
        print(f"  - 缺失海报: {len(missing_posters)} 张")

        # 保存特征
        self.save_features(poster_features)

        return poster_features, missing_posters

    def save_features(self, features):
        """保存特征到文件"""
        features_file = os.path.join(self.output_dir, "poster_features.pkl")
        with open(features_file, "wb") as f:
            pickle.dump(features, f)
        print(f"特征已保存到: {features_file}")

    def load_features(self):
        """从文件加载特征"""
        features_file = os.path.join(self.output_dir, "poster_features.pkl")
        if os.path.exists(features_file):
            with open(features_file, "rb") as f:
                return pickle.load(f)
        return None

    def compute_poster_similarity(self, movie_id, poster_features, top_k=10):
        """
        计算指定电影与所有其他电影的poster相似度

        Args:
            movie_id: 目标电影ID
            poster_features: 所有电影的poster特征字典
            top_k: 返回top_k个最相似的电影

        Returns:
            similar_movies: [(movie_id, similarity), ...]
        """
        if movie_id not in poster_features:
            return []

        target_features = poster_features[movie_id]
        similarities = []

        for other_id, features in poster_features.items():
            if other_id != movie_id:
                # 余弦相似度
                sim = np.dot(target_features, features) / (
                    np.linalg.norm(target_features) * np.linalg.norm(features)
                )
                similarities.append((other_id, sim))

        # 排序返回top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


def build_poster_database(poster_dir, output_dir, movies_df):
    """
    构建海报数据库

    Args:
        poster_dir: 海报图片目录
        output_dir: 输出目录
        movies_df: 电影DataFrame

    Returns:
        movie_id_to_path: 电影ID到海报路径的映射
    """
    print("构建海报数据库...")

    # 常见的海报文件名格式
    poster_formats = [
        "{}.jpg",  # 直接ID: "1.jpg"
        "{}.png",  # PNG格式
        "ml-1m-{}.jpg",  # 前缀格式
        "movie_{}.jpg",  # movie_前缀
    ]

    movie_id_to_path = {}
    found_count = 0

    for movie_id in movies_df["movie_id"]:
        found = False
        for fmt in poster_formats:
            path = os.path.join(poster_dir, fmt.format(movie_id))
            if os.path.exists(path):
                movie_id_to_path[movie_id] = path
                found = True
                found_count += 1
                break

        # 也尝试从目录中的文件匹配
        if not found:
            for filename in os.listdir(poster_dir):
                if filename.endswith((".jpg", ".png", ".jpeg")):
                    # 文件名可能包含movie_id
                    name_without_ext = os.path.splitext(filename)[0]
                    if str(movie_id) in name_without_ext:
                        movie_id_to_path[movie_id] = os.path.join(poster_dir, filename)
                        found = True
                        found_count += 1
                        break

    print(
        f"找到 {found_count}/{len(movies_df)} 张海报 ({found_count / len(movies_df) * 100:.1f}%)"
    )

    # 保存映射
    mapping_file = os.path.join(output_dir, "poster_mapping.pkl")
    with open(mapping_file, "wb") as f:
        pickle.dump(movie_id_to_path, f)

    return movie_id_to_path


def main():
    """主函数"""
    # 配置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    poster_dir = os.path.join(data_dir, "posters")
    output_dir = os.path.join(data_dir, "processed")

    # 加载电影数据
    movies_df = pd.read_csv(os.path.join(data_dir, "processed", "movies.csv"))

    # 构建海报数据库
    movie_id_to_path = build_poster_database(poster_dir, output_dir, movies_df)

    # 提取特征
    extractor = PosterFeatureExtractor(poster_dir, output_dir)
    movie_ids = list(movie_id_to_path.keys())
    poster_features, missing = extractor.extract_poster_features(
        movie_ids, movie_id_to_path
    )

    print(f"\n海报特征提取完成！")
    print(f"缺失率: {len(missing) / len(movie_ids) * 100:.1f}% (低于50%目标)")


if __name__ == "__main__":
    import pandas as pd

    main()
