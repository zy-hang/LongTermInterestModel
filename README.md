
# 用户长期兴趣建模

本仓库实现了一个基于 Transformer 的用户长期兴趣离线表征模型
---

## 📋 目录

- [背景](#背景)
- [数据](#数据)
- [目录结构](#目录结构)
- [环境与依赖](#环境与依赖)
- [设置与数据预处理](#设置与数据预处理)
- [模型架构](#模型架构)
- [训练](#训练)
- [推理与 Redis 集成](#推理与-redis-集成)
- [使用示例](#使用示例)
- [配置](#配置)
- [许可证](#许可证)

---

## 背景

在大规模推荐系统中，同时捕捉用户的长期偏好和近期兴趣对于提高推荐准确性至关重要。本项目实现了一个基于 BPR（Bayesian Personalized Ranking）损失的双塔架构：

1. **长期塔**：编码用户历史交互序列的前 90%。
2. **短期塔**：编码用户最近交互的后 10%。

通过最大化同一用户长期与短期向量的相似度、最小化与负样本的相似度来联合优化两座塔。

---

## 数据

- **来源**：阿里云天池移动推荐算法数据集
  下载地址：https://tianchi.aliyun.com/dataset/46
- **文件**：`tianchi_mobile_recommend_train_user.zip` → 解压后得到 `tianchi_mobile_recommend_train_user.csv`
- **字段说明**：
  - `user_id`：用户 ID（脱敏）
  - `item_id`：商品 ID（脱敏）
  - `behavior_type`：用户行为类型（1=浏览，2=收藏，3=加购，4=购买）
  - `user_geohash`：用户地理位置编码（可空）
  - `item_category`：商品类别（脱敏）
  - `time`：行为时间（小时级精度）

训练数据覆盖 **2014-11-18** 至 **2014-12-18** 的一个月行为，用于预测 **2014-12-19** 的购买情况。

---

## 目录结构

```
├── DATA/                   
│   └── train_user/           # 原始 CSV 数据
├── saved_models/             # 训练后模型权重
│   ├── long_term.pth         # 长期塔参数
│   └── short_term.pth        # 短期塔参数
├── scripts/                
│   └── train.py              # 端到端训练 & Redis 导出脚本
├── LongTermInterestModel.py  # 双塔 Transformer 模型定义
├── requirements.txt          # 依赖列表
└── README.md                 # 本文件
```

---

## 环境与依赖

- Python 3.7+
- PyTorch 1.7+
- pandas
- numpy
- scikit-learn
- tqdm
- redis-py
- （可选）CUDA

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 设置与数据预处理

1. **下载并解压** 天池数据，将 `tianchi_mobile_recommend_train_user.csv` 放到 `DATA/train_user/`。
2. **Label Encode**：将 `user_id`、`item_id` 转为整数索引，保留 `0` 作为 padding。
3. **序列聚合**：
   - 按时间升序排序
   - 前 90% 交互构成长期序列；后 10% 构成短期序列
4. **Pad/Truncate**：
   - 长期序列长度 `N_MAX`（默认 100）
   - 短期序列长度 `K_MAX`（默认 10）

上述逻辑在 `scripts/train.py` 中实现。

---

## 模型架构

- **共享 Item Embedding**：`nn.Embedding(num_items, EMBEDDING_DIM)`
- **双塔 Transformer**：各 `depth` 层多头自注意力，输出后做 mean pooling
- **输出向量维度**：`HIDDEN_DIM`

长期塔和短期塔各自一个 Transformer 编码器，但共享同一 Embedding。

---

## 训练

在 `scripts/train.py` 中运行：

```bash
python scripts/train.py \
  --data_path DATA/train_user \
  --save_dir saved_models \
  --epochs 5 \
  --batch_size 256 \
  --lr 1e-3
```

- **损失**：BPR 损失
- **优化器**：Adam
- **监控**：平均 BPR loss（可扩展为 Hit Rate、NDCG 等）

---

## 推理与 Redis 集成

训练结束后，生成每个用户的长期向量并写入 Redis：

1. 加载 `long_term.pth`。
2. 将用户的完整序列 pad/trunc 到 `N_MAX`，前向获取 128 维向量。
3. 使用原始 `user_id` 作为 key，JSON 序列化后存入 Redis。

```python
import redis, json
r = redis.Redis(host='localhost', port=6379, db=0)
for u_idx, seq in user_seqs.items():
    emb = model_long(pad(seq)).cpu().tolist()
    uid = le_user.inverse_transform([u_idx])[0]
    r.set(uid, json.dumps(emb))
```

---

## 使用示例

```bash
# 训练并导出至 Redis
python scripts/train.py --data_path DATA/train_user --save_dir saved_models

# 在线查询用户向量
>>> import redis, json
>>> r = redis.Redis()
>>> vec = json.loads(r.get('某用户ID'))
>>> # vec 即该用户的长期兴趣向量
```

---

## 配置

所有超参和路径可在 `scripts/train.py` 文件顶部修改，或通过以下命令行参数覆盖：

- `--data_path`
- `--save_dir`
- `--batch_size`
- `--epochs`
- `--lr`

---


## 许可证

本项目基于 MIT 许可证，详见 [LICENSE](LICENSE) 文件。
