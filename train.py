import os
import json
import random
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from LongTermInterestModel import *  # 你的模块
from DeepSeekMoE import MoE, ModelArgs
import redis

# -------------------
# 1. 全局配置
# -------------------
DATA_PATH = "./DATA/train_user"
CSV_FILE = "tianchi_mobile_recommend_train_user.csv"
MODEL_DIR = "./saved_models"
ITEM_EMBEDDING_DIM = 128  # item embedding 维度
CATE_EMBEDDING_DIM = 64  # 商品类目embedding维度
WEEK_EMBEDDING_DIM = 3  # 星期embedding维度
HOUR_EMBEDDING_DIM = 5  # 小时embedding维度
BEHAVIOR_EMBEDDING_DIM = 8  # 行为类型embedding维度
WEEKEND_EMBEDDING_DIM = 1  # 周末特征维度

# 特征总维度 = 商品ID(128) + 类目(64) + 星期(3) + 小时(5) + 行为类型(8) + 周末(1) + 时间戳差值(2) = 211
TOTAL_EMBEDDING_DIM = ITEM_EMBEDDING_DIM + CATE_EMBEDDING_DIM + WEEK_EMBEDDING_DIM + HOUR_EMBEDDING_DIM + BEHAVIOR_EMBEDDING_DIM + WEEKEND_EMBEDDING_DIM + 2

HIDDEN_DIM = 128  # 模型输出维度
DEPTH = 4
HEADS = 8
DIM_HEAD = 64
DROPOUT = 0.1
N_MIN = 100  # 长期序列最小长度，不足则右侧 pad
K_MIN = 10  # 短期序列最小长度，不足则右侧 pad

BATCH_SIZE = 256
EPOCHS = 5
LR = 1e-3
SEED = 42

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# MoE配置
MOE_EXPERTS = 10
MOE_ACTIVATED = 2
MOE_HIDDEN = 512
MOE_DROPOUT = 0.1


# 固定随机种子
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed()

# -------------------
# 2. 读入并预处理
# -------------------
print("1) 读取 CSV...")
df = pd.read_csv(os.path.join(DATA_PATH, CSV_FILE), parse_dates=["time"], infer_datetime_format=True)

print("2) Label Encode item_id, user_id, item_category...")
le_item = LabelEncoder()
df["item_idx"] = le_item.fit_transform(df["item_id"]) + 1  # 留 0 给 pad
n_items = df["item_idx"].nunique() + 1

le_user = LabelEncoder()
df["user_idx"] = le_user.fit_transform(df["user_id"])
n_users = df["user_idx"].nunique()

le_cate = LabelEncoder()
df["category_idx"] = le_cate.fit_transform(df["item_category"]) + 1  # 留 0 给 pad
n_categories = df["category_idx"].nunique() + 1

print(f"   用户数: {n_users}, 物品数: {n_items - 1}, 类目数: {n_categories - 1}")

# 3) 时间特征提取
print("3) 提取时间特征...")
# 星期几 (0-6)
df["weekday"] = df["time"].dt.weekday
# 是否周末 (0, 1)
df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x >= 5 else 0)
# 小时 (0-23)
df["hour"] = df["time"].dt.hour

# 4) 时间戳归一化
print("4) 处理时间戳特征...")
# 计算相对时间戳（距离最早时间的天数）
min_time = df["time"].min()
max_time = df["time"].max()
df["days_from_start"] = (df["time"] - min_time).dt.total_seconds() / (24 * 3600)
# 归一化为0-1
scaler = MinMaxScaler()
df["days_normalized"] = scaler.fit_transform(df[["days_from_start"]])
# 从今天算起的时间差值
df["days_to_end"] = (max_time - df["time"]).dt.total_seconds() / (24 * 3600)
df["days_to_end_normalized"] = scaler.fit_transform(df[["days_to_end"]])

print("5) 按 user, time 排序并构建行为序列...")
# 为了之后能够方便地构建序列，将所有特征合并为一个元组
df["feature_tuple"] = list(zip(
    df["item_idx"],
    df["category_idx"],
    df["weekday"],
    df["hour"],
    df["behavior_type"],
    df["is_weekend"],
    df["days_normalized"],
    df["days_to_end_normalized"]
))

# 按用户和时间排序，然后按用户分组
user_groups = df.sort_values("time").groupby("user_idx")["feature_tuple"].agg(list)
print(f"   构建完毕，样本用户数：{len(user_groups)}")

# 过滤掉交互过少的用户
user_seqs = {
    u: seq for u, seq in user_groups.items()
    if len(seq) >= 20  # 至少要有20条交互记录
}

print(f"   过滤后的用户数：{len(user_seqs)}")


# -------------------
# 3. 自定义 Dataset
# -------------------
class BPRDataset(Dataset):
    def __init__(self, user_seqs, n_max=N_MIN, k_max=K_MIN):
        self.users = list(user_seqs.keys())
        self.seqs = user_seqs
        self.n_max = n_max
        self.k_max = k_max

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        seq = self.seqs[u]
        cut = max(1, int(len(seq) * 0.9))
        la = seq[:cut]  # 前90%作为长期序列
        sa = seq[cut:]  # 后10%作为短期序列

        # 负采样：随机选另一个用户的 short
        while True:
            u_neg = random.choice(self.users)
            if u_neg != u:
                sb = self.seqs[u_neg][int(len(self.seqs[u_neg]) * 0.9):]
                if len(sb) > 0:
                    break

        # pad / truncate
        def pad_trunc(x, max_len):
            if len(x) >= max_len:
                return x[-max_len:]  # 取最近的max_len个
            else:
                # 用空元组填充
                pad_tuple = (0, 0, 0, 0, 0, 0, 0.0, 0.0)
                return x + [pad_tuple] * (max_len - len(x))

        la = pad_trunc(la, self.n_max)
        sa = pad_trunc(sa, self.k_max)
        sb = pad_trunc(sb, self.k_max)

        # 将元组列表转换为张量
        def tuple_to_tensors(seq):
            # 解包所有元组
            items, categories, weekdays, hours, behaviors, is_weekends, days_norm, days_to_end = zip(*seq)

            return {
                'items': torch.LongTensor(items),
                'categories': torch.LongTensor(categories),
                'weekdays': torch.LongTensor(weekdays),
                'hours': torch.LongTensor(hours),
                'behaviors': torch.LongTensor(behaviors),
                'is_weekends': torch.FloatTensor(is_weekends),
                'days_norm': torch.FloatTensor(days_norm),
                'days_to_end': torch.FloatTensor(days_to_end)
            }

        return tuple_to_tensors(la), tuple_to_tensors(sa), tuple_to_tensors(sb)


# Collate 函数需要处理字典
def collate_fn(batch):
    la, sa, sb = zip(*batch)

    # 合并每个字段
    def merge_batch_dict(batch_dicts):
        result = {}
        for key in batch_dicts[0].keys():
            result[key] = torch.stack([d[key] for d in batch_dicts])
        return result

    return merge_batch_dict(la), merge_batch_dict(sa), merge_batch_dict(sb)


# -------------------
# 4. 特征嵌入层
# -------------------
class FeatureEmbedding(nn.Module):
    def __init__(
            self,
            n_items,
            n_categories,
            item_dim=ITEM_EMBEDDING_DIM,
            cate_dim=CATE_EMBEDDING_DIM,
            week_dim=WEEK_EMBEDDING_DIM,
            hour_dim=HOUR_EMBEDDING_DIM,
            behavior_dim=BEHAVIOR_EMBEDDING_DIM,
            weekend_dim=WEEKEND_EMBEDDING_DIM,
    ):
        super().__init__()

        # 特征嵌入层
        self.item_embedding = nn.Embedding(n_items, item_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(n_categories, cate_dim, padding_idx=0)
        self.weekday_embedding = nn.Embedding(7, week_dim)  # 0-6表示周一到周日
        self.hour_embedding = nn.Embedding(24, hour_dim)  # 0-23表示一天中的小时
        self.behavior_embedding = nn.Embedding(5, behavior_dim, padding_idx=0)  # 1-4表示不同行为，0为padding

        # 合并维度，用于输出最终的特征向量
        self.total_embedding_dim = (
                item_dim + cate_dim + week_dim + hour_dim +
                behavior_dim + weekend_dim + 2  # +2是两个连续的时间特征
        )

    def forward(self, features):
        """
        参数:
            features: 包含各个特征的字典
        返回:
            合并后的嵌入向量 [batch_size, seq_len, total_embedding_dim]
        """
        # 获取各个特征的嵌入
        item_emb = self.item_embedding(features['items'])  # [B, L, item_dim]
        cate_emb = self.category_embedding(features['categories'])  # [B, L, cate_dim]
        weekday_emb = self.weekday_embedding(features['weekdays'])  # [B, L, week_dim]
        hour_emb = self.hour_embedding(features['hours'])  # [B, L, hour_dim]
        behavior_emb = self.behavior_embedding(features['behaviors'])  # [B, L, behavior_dim]

        # 连续特征处理
        is_weekend = features['is_weekends'].unsqueeze(-1)  # [B, L, 1]
        days_norm = features['days_norm'].unsqueeze(-1)  # [B, L, 1]
        days_to_end = features['days_to_end'].unsqueeze(-1)  # [B, L, 1]

        # 将所有特征拼接在一起
        concat_emb = torch.cat([
            item_emb, cate_emb, weekday_emb, hour_emb, behavior_emb,
            is_weekend, days_norm, days_to_end
        ], dim=-1)  # [B, L, total_embedding_dim]

        return concat_emb


# -------------------
# 5. 构建模型和优化器
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 特征嵌入层
feature_embedding = FeatureEmbedding(n_items, n_categories).to(device)

# 创建 MoE 模型参数
moe_args = ModelArgs(
    dim=TOTAL_EMBEDDING_DIM,
    inter_dim=MOE_HIDDEN,
    moe_inter_dim=MOE_HIDDEN,
    n_routed_experts=MOE_EXPERTS,
    n_activated_experts=MOE_ACTIVATED
)

# 长短期双塔
model_long = LongTermInterestEncoder(
    dims=TOTAL_EMBEDDING_DIM,
    output_dim=HIDDEN_DIM,
    depth=DEPTH,
    heads=HEADS,
    dim_head=DIM_HEAD,
    dropout=DROPOUT
).to(device)

model_short = LongTermInterestEncoder(
    dims=TOTAL_EMBEDDING_DIM,
    output_dim=HIDDEN_DIM,
    depth=DEPTH,
    heads=HEADS,
    dim_head=DIM_HEAD,
    dropout=DROPOUT
).to(device)

# 创建 MoE 层
moe_layer = MoE(moe_args).to(device)

# 优化器
optim_all = torch.optim.Adam(
    list(feature_embedding.parameters()) +
    list(model_long.parameters()) +
    list(model_short.parameters()) +
    list(moe_layer.parameters()),
    lr=LR
)


# -------------------
# 6. BPR Loss 定义
# -------------------
def bpr_loss(pos_sim, neg_sim):
    # -log sigmoid(pos - neg)
    return -F.logsigmoid(pos_sim - neg_sim).mean()


# -------------------
# 7. 训练循环
# -------------------
dataset = BPRDataset(user_seqs)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)

for epoch in range(1, EPOCHS + 1):
    model_long.train()
    model_short.train()
    feature_embedding.train()
    moe_layer.train()

    # 在每个 epoch 开始时重置 MoE 专家计数
    moe_layer.reset_counts()

    total_loss = 0

    with tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}") as pbar:
        for la, sa, sb in pbar:
            # 将所有特征移到设备上
            la = {k: v.to(device) for k, v in la.items()}
            sa = {k: v.to(device) for k, v in sa.items()}
            sb = {k: v.to(device) for k, v in sb.items()}

            # 特征嵌入处理
            la_emb = feature_embedding(la)  # [B, N, D]
            sa_emb = feature_embedding(sa)  # [B, K, D]
            sb_emb = feature_embedding(sb)  # [B, K, D]

            # 经过 MoE 层处理
            la_emb = moe_layer(la_emb)
            sa_emb = moe_layer(sa_emb)
            sb_emb = moe_layer(sb_emb)

            # 双塔前向传播
            u_long = model_long(la_emb)  # [B, H]
            u_pos = model_short(sa_emb)  # [B, H]
            u_neg = model_short(sb_emb)  # [B, H]

            # 余弦相似度
            pos_sim = F.cosine_similarity(u_long, u_pos, dim=-1)
            neg_sim = F.cosine_similarity(u_long, u_neg, dim=-1)

            # 计算损失
            loss = bpr_loss(pos_sim, neg_sim)

            # 反向传播
            optim_all.zero_grad()
            loss.backward()
            optim_all.step()

            total_loss += loss.item() * la['items'].size(0)
            pbar.set_postfix(loss=total_loss / ((pbar.n + 1) * BATCH_SIZE))

    # 在每个 epoch 结束时更新专家使用百分比，用于下一个 epoch 的负载均衡
    moe_layer.update_expert_percentages()

    # 打印专家使用情况
    expert_usage = moe_layer.gate.expert_percentages.cpu().numpy()
    expert_info = ", ".join([f"E{i}: {p:.2%}" for i, p in enumerate(expert_usage)])
    print(f"专家使用分布: {expert_info}")

    print(f"Epoch {epoch} 完成, 平均 Loss = {total_loss / len(dataset):.4f}")
    
    # -------------------
    # 8. 离线生成并写入 Redis
    # -------------------
    print("生成所有用户的长期兴趣 embedding 并写入 Redis...")
    model_long.eval()
    feature_embedding.eval()
    moe_layer.eval()
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    # 确保模型目录存在
    os.makedirs(MODEL_DIR, exist_ok=True)
    with torch.no_grad():
        for u, seq in tqdm(user_seqs.items()):
            # 将序列转换为模型输入格式
            padded_seq = seq + [(0, 0, 0, 0, 0, 0, 0.0, 0.0)] * (N_MIN - len(seq)) if len(seq) < N_MIN else seq[-N_MIN:]
            items, categories, weekdays, hours, behaviors, is_weekends, days_norm, days_to_end = zip(*padded_seq)
            # 准备输入特征
            features = {
                'items': torch.LongTensor(items).unsqueeze(0).to(device),
                'categories': torch.LongTensor(categories).unsqueeze(0).to(device),
                'weekdays': torch.LongTensor(weekdays).unsqueeze(0).to(device),
                'hours': torch.LongTensor(hours).unsqueeze(0).to(device),
                'behaviors': torch.LongTensor(behaviors).unsqueeze(0).to(device),
                'is_weekends': torch.FloatTensor(is_weekends).unsqueeze(0).to(device),
                'days_norm': torch.FloatTensor(days_norm).unsqueeze(0).to(device),
                'days_to_end': torch.FloatTensor(days_to_end).unsqueeze(0).to(device)
            }
            # 特征嵌入
            seq_emb = feature_embedding(features)  # [1, N, D]

            # 通过 MoE 层处理
            seq_emb = moe_layer(seq_emb)

            # 获取长期兴趣向量
            u_emb = model_long(seq_emb).squeeze(0).cpu().numpy()  # [H]
            # 存入Redis：以用户原始ID为key，值为JSON列表
            raw_u = le_user.inverse_transform([u])[0]
            r.set(raw_u, json.dumps(u_emb.tolist()))
    print("全部完成，模型权重保存在", MODEL_DIR)
    # 保存模型权重
    torch.save(feature_embedding.state_dict(), os.path.join(MODEL_DIR, "feature_embedding.pth"))
    torch.save(model_long.state_dict(), os.path.join(MODEL_DIR, "long_term.pth"))
    torch.save(model_short.state_dict(), os.path.join(MODEL_DIR, "short_term.pth"))
    torch.save(moe_layer.state_dict(), os.path.join(MODEL_DIR, "moe_layer.pth"))
