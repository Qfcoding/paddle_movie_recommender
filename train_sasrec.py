#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SASRec训练脚本 - 基于参考仓库 paddorch/SASRec.paddle

使用方法:
    python train_sasrec.py --epochs 200 --batch_size 128
"""

import os
import sys
import argparse
import numpy as np
import paddle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sasrec_ref.model import SASRec
from models.sasrec_ref.data import WarpSampler
from models.sasrec_ref.utils import set_seed, data_partition
from models.sasrec_ref.train import train as train_model, save_checkpoint
from models.sasrec_ref.eval import evaluate as evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="SASRec Training")

    parser.add_argument("--data_dir", type=str, default="./data", help="数据目录")
    parser.add_argument("--save_dir", type=str, default="./models", help="模型保存目录")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=128, help="批次大小")
    parser.add_argument("--max_len", type=int, default=200, help="序列最大长度")
    parser.add_argument("--hidden_units", type=int, default=50, help="嵌入维度")
    parser.add_argument("--num_heads", type=int, default=1, help="注意力头数量")
    parser.add_argument("--num_blocks", type=int, default=2, help="Transformer块数量")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout比例")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--l2_emb", type=float, default=0.0, help="L2正则化系数")
    parser.add_argument("--val_interval", type=int, default=800, help="评估间隔(批次)")
    parser.add_argument(
        "--val_start_batch", type=int, default=8000, help="首次评估批次"
    )
    parser.add_argument(
        "--optimizer", type=str, default="AdamW", help="优化器: Adam|AdamW|Adagrad"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["cpu", "gpu"],
        help="训练设备: cpu 或 gpu",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU设备ID (当device=gpu时有效)",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--test", action="store_true", help="仅测试模式")

    return parser.parse_args()


class Args:
    """将argparse转换为对象，用于兼容参考仓库代码"""

    def __init__(self, args):
        self.hidden_units = args.hidden_units
        self.maxlen = args.max_len
        self.dropout = args.dropout
        self.num_blocks = args.num_blocks
        self.num_heads = args.num_heads
        self.lr = args.lr
        self.l2_emb = args.l2_emb
        self.optimizer = args.optimizer
        self.save_folder = args.save_dir
        self.val_interval = args.val_interval
        self.val_start_batch = args.val_start_batch
        self.checkpoint = True
        self.save_interval = 20
        self.log_result = True
        self.continue_from = ""
        self.epochs = args.epochs
        self.batch_size = args.batch_size


def train(args):
    print("=" * 60)
    print("SASRec模型训练 (基于参考仓库 paddorch/SASRec.paddle)")
    print("=" * 60)

    set_seed(42)

    try:
        if args.device == "gpu" and paddle.is_compiled_with_cuda():
            paddle.set_device(f"gpu:{args.gpu_id}")
            print(f"使用GPU设备: {args.gpu_id}")
        else:
            paddle.set_device("cpu")
            print("使用CPU设备")
    except Exception:
        paddle.set_device("cpu")
        print("使用CPU设备")

    os.makedirs(args.save_dir, exist_ok=True)

    dataset_path = os.path.join(args.data_dir, "processed", "ratings.csv")

    if not os.path.exists(dataset_path):
        print(f"错误: 数据文件不存在 {dataset_path}")
        print("请先运行 python main.py 下载并处理数据")
        return

    processed_data_path = os.path.join(args.data_dir, "processed", "sasrec_data.txt")
    convert_ratings_to_sasrec_format(dataset_path, processed_data_path)

    dataset = data_partition(processed_data_path)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    num_batch = len(user_train) // args.batch_size
    print(f"\n数据集信息:")
    print(f"  用户数量: {usernum}")
    print(f"  物品数量: {itemnum}")
    print(f"  批次数/epoch: {num_batch}")

    seq_len = sum(len(user_train[u]) for u in user_train)
    print(f"  平均序列长度: {seq_len / len(user_train):.2f}")

    model_args = Args(args)
    model = SASRec(itemnum, model_args)

    print(f"\n模型信息:")
    print(f"  嵌入维度: {args.hidden_units}")
    print(f"  注意力头: {args.num_heads}")
    print(f"  Transformer块: {args.num_blocks}")
    print(f"  序列最大长度: {args.max_len}")

    if not args.test:
        sampler = WarpSampler(
            user_train,
            usernum,
            itemnum,
            batch_size=args.batch_size,
            maxlen=args.max_len,
            n_workers=args.num_workers,
        )
        train_model(sampler, model, model_args, num_batch, dataset)
        sampler.close()
    else:
        print("测试模式")
        evaluate_model(dataset, model, 0, 0, model_args, is_val=False)


def convert_ratings_to_sasrec_format(ratings_path, output_path):
    """将ratings.csv转换为SASRec格式 (user_id item_id 按时间排序)"""
    if os.path.exists(output_path):
        return

    print(f"转换数据格式: {ratings_path} -> {output_path}")

    import pandas as pd

    df = pd.read_csv(ratings_path)
    df = df.sort_values(["user_id", "timestamp"])

    with open(output_path, "w") as f:
        for _, row in df.iterrows():
            f.write(f"{row['user_id']} {row['movie_id']}\n")

    print(f"  转换完成: {len(df)} 条记录")


if __name__ == "__main__":
    args = parse_args()
    train(args)
