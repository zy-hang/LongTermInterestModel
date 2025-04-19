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

from sklearn.preprocessing import LabelEncoder
from LongTermInterestModel import *  # 你的模块
import redis

# -------------------
# 1. 全局配置
# -------------------
DATA_PATH = "./DATA/train_user"
CSV_FILE = "tianchi_mobile_recommend_train_user.csv"
MODEL_DIR = "./saved_models"
EMBEDDING_DIM = 64  # item embedding 维度
HIDDEN_DIM = 128  # 模型输出维度
DEPTH = 4
HEADS = 8
DIM_HEAD = 64
DROPOUT = 0.1
N_MAX = 100  # 长期序列最大长度，不足则右侧 pad
K_MAX = 10  # 短期序列最大长度，不足则右侧 pad

BATCH_SIZE = 256
EPOCHS = 5
LR = 1e-3
SEED = 42


REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0


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

print("2) Label Encode item_id, user_id...")
le_item = LabelEncoder()
df["item_idx"] = le_item.fit_transform(df["item_id"]) + 1  # 留 0 给 pad
n_items = df["item_idx"].nunique() + 1

le_user = LabelEncoder()
df["user_idx"] = le_user.fit_transform(df["user_id"])
n_users = df["user_idx"].nunique()

print(f"   用户数: {n_users}, 物品数: {n_items - 1}")

print("3) 按 user, time 排序并聚合序列...")
user_groups = df.sort_values("time").groupby("user_idx")["item_idx"].agg(list)
print(f"   构建完毕，样本用户数：{len(user_groups)}")

# 过滤掉交互过少的用户
user_seqs = {
    u: seq for u, seq in user_groups.items()
    if len(seq) >= 2  # 至少要能分割出 long/short
}


# -------------------
# 3. 自定义 Dataset
# -------------------
class BPRDataset(Dataset):
    def __init__(self, user_seqs, n_max=N_MAX, k_max=K_MAX):
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
        la = seq[:cut]
        sa = seq[cut:]

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
                return x[-max_len:]
            else:
                return x + [0] * (max_len - len(x))

        la = pad_trunc(la, self.n_max)
        sa = pad_trunc(sa, self.k_max)
        sb = pad_trunc(sb, self.k_max)

        return torch.LongTensor(la), torch.LongTensor(sa), torch.LongTensor(sb)


# Collate 自动在 DataLoader 中拼 batch
def collate_fn(batch):
    la, sa, sb = zip(*batch)
    return (
        torch.stack(la, dim=0),
        torch.stack(sa, dim=0),
        torch.stack(sb, dim=0),
    )


# -------------------
# 4. 构建模型和优化器
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# item embedding 层
item_embedding = nn.Embedding(n_items, EMBEDDING_DIM, padding_idx=0)
# 长短期双塔
model_long = LongTermInterestEncoder(dims=EMBEDDING_DIM,
                                     depth=DEPTH, heads=HEADS,
                                     dim_head=DIM_HEAD,
                                     dropout=DROPOUT).to(device)
model_short = LongTermInterestEncoder(dims=EMBEDDING_DIM,
                                      depth=DEPTH, heads=HEADS,
                                      dim_head=DIM_HEAD,
                                      dropout=DROPOUT).to(device)

optim_all = torch.optim.Adam(
    list(item_embedding.parameters()) +
    list(model_long.parameters()) +
    list(model_short.parameters()),
    lr=LR
)


# -------------------
# 5. BPR Loss 定义
# -------------------
def bpr_loss(pos_sim, neg_sim):
    # -log sigmoid(pos - neg)
    return -F.logsigmoid(pos_sim - neg_sim).mean()


# -------------------
# 6. 训练循环
# -------------------
dataset = BPRDataset(user_seqs)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)

for epoch in range(1, EPOCHS + 1):
    model_long.train()
    model_short.train()
    total_loss = 0
    with tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}") as pbar:
        for la, sa, sb in pbar:
            la = la.to(device)  # [B, N]
            sa = sa.to(device)  # [B, K]
            sb = sb.to(device)

            # embedding lookup -> [B, N, D]
            la_emb = item_embedding(la)
            sa_emb = item_embedding(sa)
            sb_emb = item_embedding(sb)

            # 双塔前向
            u_long = model_long(la_emb)  # [B, H]
            u_pos = model_short(sa_emb)
            u_neg = model_short(sb_emb)

            # 余弦相似度
            pos_sim = F.cosine_similarity(u_long, u_pos, dim=-1)
            neg_sim = F.cosine_similarity(u_long, u_neg, dim=-1)

            loss = bpr_loss(pos_sim, neg_sim)

            optim_all.zero_grad()
            loss.backward()
            optim_all.step()

            total_loss += loss.item() * la.size(0)
            pbar.set_postfix(loss=total_loss / ((pbar.n + 1) * BATCH_SIZE))

    print(f"Epoch {epoch} 完成, 平均 Loss = {total_loss / len(dataset):.4f}")

# -------------------
# 7. 离线生成并写入 Redis
# -------------------
print("生成所有用户的长期兴趣 embedding 并写入 Redis...")
model_long.eval()
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

with torch.no_grad():
    for u, seq in tqdm(user_seqs.items()):
        # pad/trunc full seq to N_MAX
        seq = seq[-N_MAX:] if len(seq) >= N_MAX else seq + [0] * (N_MAX - len(seq))
        emb = item_embedding(torch.LongTensor(seq).unsqueeze(0).to(device))  # [1, N, D]
        u_emb = model_long(emb).squeeze(0).cpu().numpy()  # [H]
        # 存 Redis：以 user 原始 ID 为 key，值为 JSON 列表
        raw_u = le_user.inverse_transform([u])[0]
        r.set(raw_u, json.dumps(u_emb.tolist()))

print("全部完成，模型权重保存在", MODEL_DIR)
torch.save(model_long.state_dict(), os.path.join(MODEL_DIR, "long_term.pth"))
torch.save(model_short.state_dict(), os.path.join(MODEL_DIR, "short_term.pth"))
