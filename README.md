# 用户长期兴趣建模

本仓库实现了一个基于 Transformer的用户长期兴趣离线表征模型。

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

1. **长期塔**：编码用户历史交互序列，输出表示长期兴趣的embedding
2. **短期塔**：采用与长期塔相同的结构（相同维度的Embedding输出）。
3. **表征输出**：使用 Transformer 的 `[CLS]` Token 位置对应的向量作为最终用户兴趣表征（替换原先的 mean pooling）。
4. **优化目标**：通过 BPR loss 构造三元组（同一用户的长/短期序列 + 负样本短期序列），最大化同一用户长短期向量的余弦相似度，最小化与负样本的相似度。

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
│   ├── feature_embedding.pth # 特征嵌入层权重
│   ├── long_term.pth         # 长期塔模型权重
│   ├── short_term.pth        # 短期塔模型权重
│   └── moe_layer.pth         # MoE层模型权重
├── train.py                  # 端到端训练 & Redis 导出脚本
├── LongTermInterestModel.py  # 双塔 Transformer 模型定义
├── DeepSeekMoE.py            # 带负载均衡的混合专家网络实现
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
4. **Padding**
   - 长期序列长度下限 `N_MIN=100`
   - 短期序列长度下限 `K_MIN=10`
5. **特征提取**
   - 商品ID编码
   - 商品类别编码
   - 时间特征：星期、小时、周末标记
   - 时间戳归一化：距离开始和结束的相对时间
   - 行为类型编码

上述逻辑已在 `scripts/train.py` 中实现。

---

## 模型架构

- **特征嵌入层**：
  - 商品嵌入 `nn.Embedding(num_items, ITEM_EMBEDDING_DIM)`
  - 类别嵌入 `nn.Embedding(num_categories, CATE_EMBEDDING_DIM)`
  - 时间特征嵌入（星期、小时、周末）
  - 行为类型嵌入
  - 连续特征投影
- **双塔 Transformer**：
  - **层数**： configurable, 默认 4 层
  - **多头自注意力**：8头注意力机制
  - **头维度**：64维
- **MoE 层（DeepSeekV3版本）**：
  - **专家数量**：可配置，默认 100 个路由专家 + 1 个共享专家
  - **激活专家数**：每个token仅激活少量专家(top-k)，默认 k=2
  - **专家结构**：包含layernorm、SwiGLU激活和dropout的前馈网络
  - **门控机制**：使用线性层+sigmoid实现的门控网络
  - **负载均衡**：
    - 每个训练epoch统计专家激活次数
    - 计算专家激活百分比
    - 在路由选择时对高使用率专家进行惩罚（除以激活百分比）
- **输出表征**：取 Transformer 输出，维度 `HIDDEN_DIM=128`

---

## 训练

```bash
python scripts/train.py \
  --data_path DATA/train_user \
  --save_dir saved_models \
  --epochs 5 \            
  --batch_size 256 \
  --lr 1e-3
```

- **负载均衡流程**：
  1. 每个epoch开始时重置专家激活计数
  2. 训练过程中累计记录专家激活次数
  3. 每个epoch结束时计算各专家激活百分比
  4. 下一个epoch使用这些百分比调整路由选择
  5. 输出专家使用分布，用于监控均衡效果
- **损失**：BPR loss（三元组）
- **优化器**：Adam
- **学习率**：固定 1e-3
- **监控指标**：平均 BPR loss 和专家使用分布

---

## 推理与 Redis 集成

1. 加载所有模型权重：
   ```python
   feature_embedding.load_state_dict(torch.load('saved_models/feature_embedding.pth'))
   model_long.load_state_dict(torch.load('saved_models/long_term.pth'))
   moe_layer.load_state_dict(torch.load('saved_models/moe_layer.pth'))
   model_long.eval()
   feature_embedding.eval()
   moe_layer.eval()
   ```
2. 构造全序列（pad/trunc 至 `N_MAX`），通过模型计算向量：
   ```python
   # 特征嵌入
   seq_emb = feature_embedding(features)
   # 通过MoE层
   seq_emb = moe_layer(seq_emb)
   # 获取长期兴趣向量
   u_emb = model_long(seq_emb).cpu().tolist()
   ```
3. 将脱敏后的 `user_id` 反编码为原始 ID，JSON 序列化后写入 Redis：
   ```python
   import redis, json
   r = redis.Redis(host='localhost', port=6379, db=0)
   r.set(raw_user_id, json.dumps(u_emb))
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

- **数据相关**
  - `--data_path`
  - `--save_dir`
- **训练相关**
  - `--batch_size`
  - `--epochs`
  - `--lr`
- **模型相关**
  - `--item_embedding_dim`
  - `--hidden_dim`
  - `--depth`
  - `--heads`
- **MoE相关**
  - `--moe_experts`: 专家数量
  - `--moe_activated`: 激活专家数
  - `--moe_hidden`: 专家隐藏层维度

---

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE)。

---
