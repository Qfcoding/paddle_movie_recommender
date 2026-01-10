#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强评估指标模块
提供完整的推荐系统评估指标：RMSE, MAE, ACC, 覆盖率, 多样性, 混淆矩阵等
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算均方根误差 (Root Mean Square Error)

    Args:
        y_true: 真实评分
        y_pred: 预测评分

    Returns:
        RMSE值
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算平均绝对误差 (Mean Absolute Error)

    Args:
        y_true: 真实评分
        y_pred: 预测评分

    Returns:
        MAE值
    """
    return np.mean(np.abs(y_true - y_pred))


def compute_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 3.5
) -> float:
    """
    计算准确率 (Accuracy) - 二分类版本
    将评分二值化: >= threshold 为正类(喜欢), < threshold 为负类(不喜欢)

    Args:
        y_true: 真实评分
        y_pred: 预测评分
        threshold: 分类阈值

    Returns:
        准确率 (0-1)
    """
    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)

    correct = np.sum(y_true_binary == y_pred_binary)
    total = len(y_true_binary)

    return correct / total if total > 0 else 0.0


def compute_precision_recall_at_k(
    recommendations: Dict[int, List[int]],
    ground_truth: Dict[int, List[int]],
    k: int = 10,
) -> Tuple[float, float]:
    """
    计算Precision@K 和 Recall@K

    Args:
        recommendations: 用户ID到推荐列表的映射
        ground_truth: 用户ID到真实交互列表的映射
        k: 推荐列表长度

    Returns:
        (Precision@K, Recall@K)
    """
    precisions = []
    recalls = []

    for user_id, recs in recommendations.items():
        if user_id not in ground_truth:
            continue

        true_items = set(ground_truth[user_id])
        rec_items = set(recs[:k])

        if len(rec_items) > 0:
            precision = len(true_items & rec_items) / len(rec_items)
            precisions.append(precision)

        if len(true_items) > 0:
            recall = len(true_items & rec_items) / len(true_items)
            recalls.append(recall)

    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_recall = np.mean(recalls) if recalls else 0.0

    return avg_precision, avg_recall


def compute_hit_rate(
    recommendations: Dict[int, List[int]], ground_truth: Dict[int, List[int]]
) -> float:
    """
    计算命中率 (Hit Rate)
    至少有一个推荐项在真实交互中的用户比例

    Args:
        recommendations: 用户ID到推荐列表的映射
        ground_truth: 用户ID到真实交互列表的映射

    Returns:
        Hit Rate (0-1)
    """
    hits = 0
    total = 0

    for user_id, recs in recommendations.items():
        if user_id not in ground_truth:
            continue

        total += 1
        true_items = set(ground_truth[user_id])
        if len(set(recs) & true_items) > 0:
            hits += 1

    return hits / total if total > 0 else 0.0


def compute_coverage(
    recommendations: Dict[int, List[int]], all_items: List[int]
) -> float:
    """
    计算覆盖率 (Coverage)
    推荐系统中被推荐过的物品占总物品的比例

    Args:
        recommendations: 用户ID到推荐列表的映射
        all_items: 所有物品ID列表

    Returns:
        覆盖率 (0-1)
    """
    recommended_items = set()
    for recs in recommendations.values():
        recommended_items.update(recs)

    return len(recommended_items) / len(all_items) if all_items else 0.0


def compute_diversity(
    recommendations: Dict[int, List[int]],
    similarity_matrix: Dict[int, Dict[int, float]],
) -> float:
    """
    计算多样性 (Diversity)
    1 - 平均相似度

    Args:
        recommendations: 用户ID到推荐列表的映射
        similarity_matrix: 物品相似度矩阵

    Returns:
        多样性值 (0-1, 越高越多样)
    """
    diversities = []

    for user_id, recs in recommendations.items():
        if len(recs) < 2:
            continue

        similarities = []
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                item_i, item_j = recs[i], recs[j]
                if item_i in similarity_matrix and item_j in similarity_matrix[item_i]:
                    sim = similarity_matrix[item_i][item_j]
                    similarities.append(sim)

        if similarities:
            avg_sim = np.mean(similarities)
            diversity = 1 - avg_sim
            diversities.append(diversity)

    return np.mean(diversities) if diversities else 0.0


def compute_novelty(
    recommendations: Dict[int, List[int]],
    item_popularity: Dict[int, int],
    total_interactions: int,
) -> float:
    """
    计算新颖性 (Novelty)
    推荐物品的罕见程度

    Args:
        recommendations: 用户ID到推荐列表的映射
        item_popularity: 物品流行度字典
        total_interactions: 总交互次数

    Returns:
        新颖性值 (越高越新颖)
    """
    novelties = []

    for user_id, recs in recommendations.items():
        for item_id in recs:
            if item_id in item_popularity:
                popularity = item_popularity[item_id]
                novelty = -np.log2(popularity / total_interactions + 1e-10)
                novelties.append(novelty)

    return np.mean(novelties) if novelties else 0.0


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 3.5
) -> np.ndarray:
    """
    计算混淆矩阵 (二分类)

    Args:
        y_true: 真实评分
        y_pred: 预测评分
        threshold: 分类阈值

    Returns:
        混淆矩阵 [[TN, FP], [FN, TP]]
    """
    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)

    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))

    return np.array([[tn, fp], [fn, tp]])


def compute_f1_score(precision: float, recall: float) -> float:
    """
    计算F1分数

    Args:
        precision: 精确率
        recall: 召回率

    Returns:
        F1分数
    """
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class EnhancedMetrics:
    """
    增强评估指标计算器
    整合所有评估指标，提供完整的评估报告
    """

    def __init__(
        self, rating_threshold: float = 3.5, k_values: List[int] = [5, 10, 20]
    ):
        """
        初始化评估器

        Args:
            rating_threshold: 评分阈值，用于二分类
            k_values: 不同的K值列表
        """
        self.rating_threshold = rating_threshold
        self.k_values = k_values

    def full_evaluation(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        recommendations: Optional[Dict[int, List[int]]] = None,
        ground_truth: Optional[Dict[int, List[int]]] = None,
        all_items: Optional[List[int]] = None,
        similarity_matrix: Optional[Dict[int, Dict[int, float]]] = None,
        item_popularity: Optional[Dict[int, int]] = None,
        total_interactions: int = 0,
    ) -> Dict:
        """
        完整评估报告

        Args:
            y_true: 真实评分
            y_pred: 预测评分
            recommendations: 推荐结果 (可选)
            ground_truth: 真实交互 (可选)
            all_items: 所有物品 (可选)
            similarity_matrix: 相似度矩阵 (可选)
            item_popularity: 物品流行度 (可选)
            total_interactions: 总交互数 (可选)

        Returns:
            包含所有指标的字典
        """
        results = {}

        # 回归指标
        results["RMSE"] = compute_rmse(y_true, y_pred)
        results["MAE"] = compute_mae(y_true, y_pred)

        # 分类指标
        results["Accuracy"] = compute_accuracy(y_true, y_pred, self.rating_threshold)

        # 混淆矩阵
        results["ConfusionMatrix"] = compute_confusion_matrix(
            y_true, y_pred, self.rating_threshold
        ).tolist()

        # 推荐指标
        if recommendations is not None and ground_truth is not None:
            # Precision@K, Recall@K
            for k in self.k_values:
                prec, rec = compute_precision_recall_at_k(
                    recommendations, ground_truth, k
                )
                results[f"Precision@{k}"] = prec
                results[f"Recall@{k}"] = rec
                results[f"F1@{k}"] = compute_f1_score(prec, rec)

            # Hit Rate
            results["HitRate"] = compute_hit_rate(recommendations, ground_truth)

            # 覆盖率
            if all_items is not None:
                results["Coverage"] = compute_coverage(recommendations, all_items)

            # 多样性
            if similarity_matrix is not None:
                results["Diversity"] = compute_diversity(
                    recommendations, similarity_matrix
                )

            # 新颖性
            if item_popularity is not None and total_interactions > 0:
                results["Novelty"] = compute_novelty(
                    recommendations, item_popularity, total_interactions
                )

        return results

    def print_report(self, results: Dict) -> None:
        """打印评估报告"""
        print("\n" + "=" * 50)
        print("推荐系统评估报告")
        print("=" * 50)

        # 回归指标
        print("\n【回归指标】")
        print(f"  RMSE: {results.get('RMSE', 'N/A'):.4f}")
        print(f"  MAE:  {results.get('MAE', 'N/A'):.4f}")

        # 分类指标
        print("\n【分类指标】")
        print(f"  Accuracy: {results.get('Accuracy', 'N/A'):.4f}")

        if "ConfusionMatrix" in results:
            cm = results["ConfusionMatrix"]
            print(f"  混淆矩阵:")
            print(f"    TN={cm[0][0]}, FP={cm[0][1]}")
            print(f"    FN={cm[1][0]}, TP={cm[1][1]}")

        # 推荐指标
        if "Precision@10" in results:
            print("\n【推荐指标】")
            print(f"  Precision@10: {results.get('Precision@10', 'N/A'):.4f}")
            print(f"  Recall@10:    {results.get('Recall@10', 'N/A'):.4f}")
            print(f"  F1@10:        {results.get('F1@10', 'N/A'):.4f}")
            print(f"  Hit Rate:     {results.get('HitRate', 'N/A'):.4f}")
            print(f"  Coverage:     {results.get('Coverage', 'N/A'):.4f}")
            print(f"  Diversity:    {results.get('Diversity', 'N/A'):.4f}")
            if "Novelty" in results:
                print(f"  Novelty:      {results.get('Novelty', 'N/A'):.4f}")

        print("\n" + "=" * 50)


def main():
    """测试代码"""
    # 模拟数据
    y_true = np.array([5, 4, 3, 2, 1, 5, 4, 3, 2, 1])
    y_pred = np.array([4.8, 4.2, 3.1, 2.3, 1.2, 4.9, 3.8, 3.5, 1.8, 1.5])

    # 测试指标
    print("测试评估指标计算:")
    print(f"RMSE: {compute_rmse(y_true, y_pred):.4f}")
    print(f"MAE:  {compute_mae(y_true, y_pred):.4f}")
    print(f"Accuracy: {compute_accuracy(y_true, y_pred):.4f}")

    # 完整评估
    evaluator = EnhancedMetrics()
    results = evaluator.full_evaluation(y_true, y_pred)
    evaluator.print_report(results)


if __name__ == "__main__":
    main()
