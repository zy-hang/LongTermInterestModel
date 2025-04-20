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
# 5. 双塔模型定义
# -------------------
class DualTowerModel(nn.Module):
    def __init__(self, n_items, n_categories, hidden_dim=HIDDEN_DIM, depth=DEPTH, heads=HEADS, dim_head=DIM_HEAD,
                 dropout=DROPOUT):
        super().__init__()
        self.feature_embedding = FeatureEmbedding(n_items, n_categories)

        # 长期兴趣塔 - 使用LongTermInterestEncoder
        self.long_tower = LongTermInterestEncoder(
            dims=TOTAL_EMBEDDING_DIM,
            output_dim=hidden_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout
        )

        # 短期兴趣塔 - 同样使用LongTermInterestEncoder架构
        self.short_tower = LongTermInterestEncoder(
            dims=TOTAL_EMBEDDING_DIM,
            output_dim=hidden_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout
        )

    def forward(self, long_features, short_features):
        # 特征嵌入
        long_emb = self.feature_embedding(long_features)  # [B, L, D]
        short_emb = self.feature_embedding(short_features)  # [B, K, D]

        # 通过各自的塔获取表征
        long_repr = self.long_tower(long_emb)  # [B, hidden_dim]
        short_repr = self.short_tower(short_emb)  # [B, hidden_dim]

        return long_repr, short_repr

    def update_moe_balancing(self):
        """更新所有MoE层的负载均衡统计"""
        self.long_tower.update_moe_balancing()
        self.short_tower.update_moe_balancing()

    def reset_moe_counts(self):
        """重置所有MoE层的专家激活计数"""
        self.long_tower.reset_moe_counts()
        self.short_tower.reset_moe_counts()


# -------------------
# 6. BPR损失函数
# -------------------
class BPRLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        计算BPR损失
        参数:
            anchor: 长期兴趣表征 [B, D]
            positive: 正样本，同一用户的短期兴趣表征 [B, D]
            negative: 负样本，其他用户的短期兴趣表征 [B, D]
        """
        pos_score = F.cosine_similarity(anchor, positive, dim=1)
        neg_score = F.cosine_similarity(anchor, negative, dim=1)

        # BPR损失: -log(sigmoid(pos_score - neg_score))
        loss = -F.logsigmoid(pos_score - neg_score).mean()

        return loss


# -------------------
# 7. 训练函数
# -------------------
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    # 使用tqdm创建进度条
    progress_bar = tqdm(train_loader)
    for la, sa, sb in progress_bar:
        # 将数据移到指定设备
        for key in la:
            la[key] = la[key].to(device)
            sa[key] = sa[key].to(device)
            sb[key] = sb[key].to(device)

        # 前向传播
        long_repr, short_repr = model(la, sa)
        _, neg_repr = model(la, sb)

        # 计算损失
        loss = criterion(long_repr, short_repr, neg_repr)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_description(f"Loss: {loss.item():.4f}")

    return total_loss / len(train_loader)


# -------------------
# 8. 验证函数
# -------------------
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for la, sa, sb in val_loader:
            # 将数据移到指定设备
            for key in la:
                la[key] = la[key].to(device)
                sa[key] = sa[key].to(device)
                sb[key] = sb[key].to(device)

            # 前向传播
            long_repr, short_repr = model(la, sa)
            _, neg_repr = model(la, sb)

            # 计算损失
            loss = criterion(long_repr, short_repr, neg_repr)
            total_loss += loss.item()

    return total_loss / len(val_loader)


# -------------------
# 9. 离线生成用户长期兴趣向量
# -------------------
def generate_user_representations(model, user_seqs, device):
    model.eval()
    user_reprs = {}

    with torch.no_grad():
        for user_id, seq in tqdm(user_seqs.items(), desc="生成用户表征"):
            if len(seq) < 20:  # 确保有足够的行为序列
                continue

            # 处理序列
            if len(seq) > N_MIN:
                feature_seq = seq[:N_MIN]  # 使用最早的N_MIN个行为
            else:
                # padding
                pad_tuple = (0, 0, 0, 0, 0, 0, 0.0, 0.0)
                feature_seq = seq + [pad_tuple] * (N_MIN - len(seq))

            # 转换为张量
            items, categories, weekdays, hours, behaviors, is_weekends, days_norm, days_to_end = zip(*feature_seq)
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

            # 仅使用长期塔获取表征
            feature_emb = model.feature_embedding(features)
            long_repr = model.long_tower(feature_emb)

            # 存储表征
            user_reprs[user_id] = long_repr.cpu().numpy().tolist()[0]

    return user_reprs


# -------------------
# 10. 将表征存入Redis
# -------------------
def save_to_redis(user_reprs):
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        # 先清空已有数据
        r.flushdb()

        # 存储用户表征
        pipe = r.pipeline()
        for user_id, repr_vector in tqdm(user_reprs.items(), desc="存储到Redis"):
            key = f"user:{user_id}"
            pipe.set(key, json.dumps(repr_vector))

        # 执行批量操作
        pipe.execute()
        print(f"成功将 {len(user_reprs)} 条用户表征存入Redis")
    except Exception as e:
        print(f"存储到Redis失败: {e}")


# -------------------
# 11. 主函数
# -------------------
def main():
    # 检查是否有GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建保存模型的目录
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 创建数据集和加载器
    dataset = BPRDataset(user_seqs)

    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )

    # 初始化模型
    model = DualTowerModel(n_items, n_categories).to(device)

    # 初始化优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = BPRLoss()

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=True
    )

    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # 训练
        train_loss = train(model, train_loader, optimizer, criterion, device)

        # 在每个epoch结束时更新MoE负载均衡统计
        model.update_moe_balancing()

        # 验证
        val_loss = validate(model, val_loader, criterion, device)

        print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")

        # 学习率调整
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))
            print("保存最佳模型")

        # 每个epoch开始重置MoE计数
        model.reset_moe_counts()

    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pth")))

    # 离线生成用户长期兴趣表征
    print("\n生成用户长期兴趣表征...")
    user_reprs = generate_user_representations(model, user_seqs, device)

    # 保存到Redis
    print("\n将用户表征存入Redis...")
    save_to_redis(user_reprs)

    # 保存模型最终版本
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "final_model.pth"))
    print("\n训练完成!")


if __name__ == "__main__":
    main()
