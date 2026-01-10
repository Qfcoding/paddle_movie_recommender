#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SASRec模型 - 基于Transformer的序列推荐

参考: https://github.com/paddorch/SASRec.paddle
论文: "Self-Attentive Sequential Recommendation", WWW 2019
"""

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class SASRec(paddle.nn.Layer):
    def __init__(
        self,
        item_num,
        max_len=50,
        hidden_units=64,
        num_heads=2,
        num_blocks=2,
        dropout_rate=0.5,
    ):
        super(SASRec, self).__init__()
        self.item_num = item_num
        self.max_len = max_len
        self.hidden_units = hidden_units

        self.item_emb = nn.Embedding(item_num + 1, hidden_units)
        self.pos_emb = nn.Embedding(max_len, hidden_units)
        self.emb_dropout = paddle.nn.Dropout(p=dropout_rate)

        self.subsequent_mask = paddle.triu(paddle.ones((max_len, max_len))) == 0

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_units,
            nhead=num_heads,
            dim_feedforward=hidden_units,
            dropout=dropout_rate,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=num_blocks
        )

    def position_encoding(self, seqs):
        seqs_embed = self.item_emb(seqs)
        positions = np.tile(np.array(range(seqs.shape[1])), [seqs.shape[0], 1])
        position_embed = self.pos_emb(paddle.to_tensor(positions, dtype="int64"))
        return self.emb_dropout(seqs_embed + position_embed)

    def forward(self, log_seqs, pos_seqs, neg_seqs):
        seqs_embed = self.position_encoding(log_seqs)
        log_feats = self.encoder(seqs_embed, self.subsequent_mask)

        pos_embed = self.item_emb(pos_seqs)
        neg_embed = self.item_emb(neg_seqs)

        pos_logits = (log_feats * pos_embed).sum(axis=-1)
        neg_logits = (log_feats * neg_embed).sum(axis=-1)

        return pos_logits, neg_logits

    def predict(self, log_seqs, item_indices):
        seqs = self.position_encoding(log_seqs)
        log_feats = self.encoder(seqs, self.subsequent_mask)

        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(paddle.to_tensor(item_indices, dtype="int64"))

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits


class MyBCEWithLogitLoss(paddle.nn.Layer):
    def __init__(self):
        super(MyBCEWithLogitLoss, self).__init__()

    def forward(self, pos_logits, neg_logits, labels):
        return paddle.sum(
            -paddle.log(F.sigmoid(pos_logits) + 1e-24) * labels
            - paddle.log(1 - F.sigmoid(neg_logits) + 1e-24) * labels,
            axis=(0, 1),
        ) / paddle.sum(labels, axis=(0, 1))


def sasrec_predict_next_items(model, user_history, item_indices, device="gpu"):
    model.eval()

    max_len = model.max_len
    if len(user_history) > max_len:
        seq = user_history[-max_len:]
    else:
        seq = [0] * (max_len - len(user_history)) + user_history

    seq = paddle.to_tensor([seq], dtype="int64")

    with paddle.no_grad():
        logits = model.predict(seq, item_indices)
        logits = logits.numpy()[0]

    sorted_indices = np.argsort(-logits)
    return logits[sorted_indices], [item_indices[i] for i in sorted_indices]


if __name__ == "__main__":
    model = SASRec(item_num=100, max_len=50, hidden_units=64)

    batch_size = 2
    log_seqs = paddle.randint(0, 100, [batch_size, 50])
    pos_seqs = paddle.randint(0, 100, [batch_size, 50])
    neg_seqs = paddle.randint(0, 100, [batch_size, 50])

    pos_logits, neg_logits = model(log_seqs, pos_seqs, neg_seqs)

    print(f"输入形状: log_seqs={log_seqs.shape}")
    print(f"正样本logits形状: {pos_logits.shape}")
    print(f"负样本logits形状: {neg_logits.shape}")
    print("✓ SASRec模型测试通过")
