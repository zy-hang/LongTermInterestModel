
---

# 用户长期兴趣建模

本仓库实现了一个基于 Transformer 的用户长期兴趣离线表征模型。

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

在大规模推荐系统中，同时捕捉用户的长期偏好和近期兴趣对于提高推荐准确性至关重要。在线对用户行为序列进行向量表征，有以下的问题：average pool的方法低效；使用双层attention的DIN模型受限于序列长度；DIEN模型为GRU隐藏状态加入辅助损失函数会增加项目部署难度；SIM模型通过hard/soft search策略排除低相关行为序列，会丢失部分有用信息。而通过离线计算用户长期兴趣的向量表征，缓存起来供在线模型索引，可兼顾开销与性能。本项目实现了一个基于 Transformer 的用户长期兴趣离线表征模型，包括：

1. **长期塔**：编码用户历史交互序列的前 90%，采用 Transformer，并引入 RoPE（相对位置编码）以捕捉序列内的相对位置信息。
2. **短期塔**：编码用户最近交互的后 10%，采用与长期塔相同的结构（共享 Item Embedding）。
3. **MoE（DeepSeek）**：在 Transformer 输出后接入带负载均衡的专家网络，实现个性化非线性变换，解耦共享参数并动态适应不同序列长度。
4. **表征输出**：使用 Transformer 的 `[CLS]` Token 位置对应的向量作为最终用户兴趣表征（替换原先的 mean pooling）。
5. **优化目标**：通过 BPR loss 构造三元组（同一用户的长/短期序列 + 负样本短期序列），最大化同一用户长短期向量的余弦相似度，最小化与负样本的相似度。

---

## 数据

- **数据集**：阿里云天池移动推荐算法公开数据集
- **下载**：https://tianchi.aliyun.com/dataset/46
- **原始文件**：
  - `tianchi_mobile_recommend_train_user.csv`
- **字段说明**：
  - `user_id`：脱敏用户 ID
  - `item_id`：脱敏商品 ID
  - `behavior_type`：行为类型（1=浏览，2=收藏，3=加购，4=购买）
  - `user_geohash`：地理位置编码（可空）
  - `item_category`：商品类别
  - `time`：行为时间（小时级精度）
- **时间范围**：2014‑11‑18 至 2014‑12‑18，用于预测 2014‑12‑19 的购买。

---

## 目录结构

```
├── DATA/                 
│   └── train_user/           # 原始 CSV 数据
├── saved_models/           
│   ├── long_term.pth         # 长期塔模型权重
│   └── short_term.pth        # 短期塔模型权重
├── scripts/              
│   └── train.py              # 端到端训练 & Redis 导出脚本
├── LongTermInterestModel.py  # 双塔 Transformer + MoE 模型定义
├── requirements.txt          # 依赖列表
└── README.md                 # 本文件
```

---

## 环境与依赖

- **Python**：3.8+
- **PyTorch**：1.9+（推荐 1.10）
- **CUDA**：11.1+（若使用 GPU）
- **其它**：
  - pandas
  - numpy
  - scikit-learn
  - tqdm
  - redis-py

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 设置与数据预处理

1. **下载并解压**
   将 `tianchi_mobile_recommend_train_user.csv` 放入 `DATA/train_user/`。
2. **Label Encode**
   将 `user_id`、`item_id` 映射为连续整数索引，保留 `0` 作为 padding。
3. **序列切分**
   - 按时间升序排序
   - 前 90% 交互构成长期序列；后 10% 构成短期序列
4. **Pad/Truncate**
   - 长期序列长度上限 `N_MAX=100`
   - 短期序列长度上限 `K_MAX=10`

上述逻辑已在 `scripts/train.py` 中实现。

---

## 模型架构

- **共享 Embedding**：`nn.Embedding(num_items, EMBEDDING_DIM)`
- **双塔 Transformer**：
  - **层数**： configurable, 默认 16 层
  - **多头自注意力**：支持 RoPE 相对位置编码
- **MoE 层**：DeepSeek 带负载均衡的专家网络
- **输出表征**：取 Transformer `[CLS]` token 对应向量，维度 `HIDDEN_DIM`

---

## 训练

```bash
python scripts/train.py \
  --data_path DATA/train_user \
  --save_dir saved_models \
  --epochs 100 \              
  --batch_size 256 \
  --lr 1e-3 \     
```

- **损失**：BPR loss（三元组）
- **优化器**：AdamW + 权重衰减
- **学习率调度**：线性 warm‑up + 余弦衰减
- **监控指标**：平均 BPR loss
- **Checkpoint**：每隔 1 epoch 执行一次验证，结合早停保存最佳模型

---

## 推理与 Redis 集成

1. 加载 `long_term.pth`：
   ```python
   model_long.load_state_dict(torch.load('saved_models/long_term.pth'))
   model_long.eval()
   ```
2. 构造全序列（pad/trunc 至 `N_MAX`），前向计算 128 维向量：
   ```python
   emb = model_long(pad_seq).cpu().tolist()
   ```
3. 将脱敏后的 `user_id` 反编码为原始 ID，JSON 序列化后写入 Redis：
   ```python
   import redis, json
   r = redis.Redis(host='localhost', port=6379, db=0)
   r.set(raw_user_id, json.dumps(emb))
   ```

---

## 使用示例

```bash
# 训练并导出至 Redis
python scripts/train.py --data_path DATA/train_user --save_dir saved_models

# 在线读取用户长期向量
>>> import redis, json
>>> r = redis.Redis()
>>> vec = json.loads(r.get('user_1234'))
>>> # vec 即该用户的 128 维长期兴趣向量
```

---

## 配置

所有超参均可在 `scripts/train.py` 顶部修改，或通过命令行参数覆盖：

- `--data_path`
- `--save_dir`
- `--batch_size`
- `--epochs`
- `--lr`
- `--warmup_steps`
- `--early_stop_patience`

---

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE)。

---

