import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.utils.prune as prune
import torchvision.utils as vutils
from einops import rearrange
from ptflops import get_model_complexity_info
import time
from thop import profile
from DeepSeekMoE import *


class RotaryPositionalEmbedding(nn.Module):
    """
    旋转位置编码 (RoPE) 模块

    这个模块可以轻松集成到现有的注意力机制中，只需在计算注意力之前
    将Q和K通过此模块处理即可。
    参数:
        dim (int): 需要应用旋转的维度大小（通常是注意力头的维度）
        max_seq_len (int): 预计算的最大序列长度
        base (float): 用于计算旋转频率的基数，默认为10000.0
        scaling_factor (float): 用于长序列外推的缩放因子，默认为1.0（不缩放）
        device (torch.device, optional): 计算设备
    """

    def __init__(
            self,
            dim: int,
            max_seq_len: int = 2048,
            base: float = 10000.0,
            scaling_factor: float = 1.0,
            device: Optional[torch.device] = None
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor

        # 计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        # 应用缩放因子
        inv_freq = inv_freq * scaling_factor
        self.register_buffer("inv_freq", inv_freq)

        # 预计算缓存
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """预计算并缓存旋转角度的正弦和余弦值"""
        self.max_seq_len_cached = seq_len

        # 计算位置索引 * 频率
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [seq_len, dim/2]

        # 计算复数的实部和虚部（余弦和正弦）
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用旋转位置编码到查询和键向量
        参数:
            q: 查询向量，形状为 [..., seq_len, dim]
            k: 键向量，形状为 [..., seq_len, dim]
        返回:
            应用了旋转位置编码的查询和键向量
        """
        seq_len = q.shape[-2]

        # 如果需要，扩展缓存
        if seq_len > self.max_seq_len_cached:
            self._build_cache(seq_len)

        cos = self.cos_cached[:seq_len]  # [seq_len, dim]
        sin = self.sin_cached[:seq_len]  # [seq_len, dim]

        # 应用旋转
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)

    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """应用旋转操作到输入张量"""
        # 确保维度匹配
        if x.ndim == 4 and cos.ndim == 2:  # 多头情况
            # [batch, heads, seq_len, dim] 和 [seq_len, dim]
            # 需要调整余弦和正弦的形状
            cos = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, dim]
            sin = sin.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, dim]

        # 将输入分成两半，对应实部和虚部
        x_half1 = x[..., :self.dim // 2]
        x_half2 = x[..., self.dim // 2:self.dim]

        # 如果维度不匹配（例如，x的维度大于self.dim），保留额外的维度
        if x.shape[-1] > self.dim:
            extra_dims = x[..., self.dim:]

            # 应用旋转（复数乘法）
            rotated_half1 = x_half1 * cos[..., :self.dim // 2] - x_half2 * sin[..., :self.dim // 2]
            rotated_half2 = x_half1 * sin[..., :self.dim // 2] + x_half2 * cos[..., :self.dim // 2]

            # 合并回原始形状
            rotated_x = torch.cat([rotated_half1, rotated_half2, extra_dims], dim=-1)
        else:
            # 应用旋转（复数乘法）
            rotated_half1 = x_half1 * cos[..., :self.dim // 2] - x_half2 * sin[..., :self.dim // 2]
            rotated_half2 = x_half1 * sin[..., :self.dim // 2] + x_half2 * cos[..., :self.dim // 2]

            # 合并回原始形状
            rotated_x = torch.cat([rotated_half1, rotated_half2], dim=-1)

        return rotated_x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, att_dropout=0., out_dropout=0.):
        super().__init__()
        self.dim = dim
        self.inner_dim = dim_head * heads
        self.project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5  # 缩放因子用于防止点积结果过大，从而导致梯度消失或爆炸的问题
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)  # 对行进行softmax
        self.dropout = nn.Dropout(att_dropout)
        self.out_dropout = nn.Dropout(out_dropout)

        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, self.dim)

    def forward(self, x):
        x = self.norm(x)
        'x为[b,n+1, 1024], 经过to_qkv后变为[b,n+1,heads * heads_dim * 3]'
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # chunk是将倒数第一个维度分为3份
        'qkv是一个有三个张量的元组，每个张量的大小都是[b,n+1,heads * head_dim]'
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        'q,k,v的大小都是[b,heads,n+1, heads_dim]'
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        '将k的倒数1、2维转置  q[b,heads, n+1, head_dim] 转置后的k[b,heads, head_dim, n+1] 结果dots[b,heads,n+1,n+1]'
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        'attn[b,heads,n+1,n+1]  v[b,heads, n+1, head_dim] 结果out[b,heads, n+1, head_dim]'
        out = rearrange(out, 'b h n d -> b n (h d)')
        'out[b,n+1,heads * head_dim]'
        out = self.to_out(out)
        out = self.out_dropout(out)
        return out


class Attention_RoPE(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, att_dropout=0., out_dropout=0., max_seq_len=2048, base=10000.0,
                 scaling_factor=1.0, device=None):
        super().__init__()
        self.dim = dim
        self.inner_dim = dim_head * heads
        self.project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5  # 缩放因子用于防止点积结果过大，从而导致梯度消失或爆炸的问题
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)  # 对行进行softmax
        self.dropout = nn.Dropout(att_dropout)
        self.out_dropout = nn.Dropout(out_dropout)
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, self.dim)
        # 添加 RoPE (Rotary Positional Encoding)
        self.rope = RotaryPositionalEmbedding(dim_head, max_seq_len=max_seq_len, base=base,
                                              scaling_factor=scaling_factor, device=device)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        q, k = self.rope(q, k)  # 应用 RoPE
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.out_dropout(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dims, depth=16, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dims)
        self.layers = nn.ModuleList([])

        # 为MoE层创建合适的配置
        moe_args = ModelArgs(
            dim=dims,
            inter_dim=dims * 4,  # 通常FFN的中间层是维度的4倍
            moe_inter_dim=dims * 4,
            n_routed_experts=10,  # 可根据需要调整
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            score_func="sigmoid",
            route_scale=1.0
        )

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention_RoPE(dim=dims, heads=heads, dim_head=dim_head, att_dropout=dropout, out_dropout=dropout),
                MoE(moe_args),  # 使用配置初始化MoE
            ]))

        # 存储所有MoE层的引用，方便更新负载均衡
        self.moe_layers = [layer[1] for layer in self.layers]

    def forward(self, x):
        for attn, ff in self.layers:
            x = x + attn(x)  # 残差连接
            x = x + ff(x)  # 残差连接

        return self.norm(x)

    def update_moe_balancing(self):
        """更新所有MoE层的负载均衡"""
        for moe in self.moe_layers:
            moe.update_expert_percentages()

    def reset_moe_counts(self):
        """重置所有MoE层的激活计数"""
        for moe in self.moe_layers:
            moe.reset_counts()


class LongTermInterestEncoder(nn.Module):
    """
    用户长期兴趣离线表征模块
    输入: 用户长期行为序列的 embedding 张量 x，形状 [batch_size, N, dims]
    输出: 归一化的用户长期兴趣向量，形状 [batch_size, output_dim]
    """

    def __init__(
            self,
            dims: int,
            output_dim: int = 128,
            depth: int = 4,
            heads: int = 8,
            dim_head: int = 64,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.dims = dims
        self.output_dim = output_dim
        # 可学习的 CLS token，用于聚合序列信息
        self.cls_token = nn.Parameter(torch.randn(1, 1, dims))
        # Transformer 编码器，内部包含 Attention_RoPE + MoE
        self.transformer = Transformer(
            dims=dims,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout
        )
        # 将 CLS 输出映射到目标维度
        self.proj = nn.Linear(dims, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def update_moe_balancing(self):
        """更新transformer中所有MoE层的负载均衡"""
        self.transformer.update_moe_balancing()

    def reset_moe_counts(self):
        """重置transformer中所有MoE层的激活计数"""
        self.transformer.reset_moe_counts()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向计算
        参数:
            x (torch.Tensor): 行为序列 embedding，[batch_size, N, dims]
        返回:
            torch.Tensor: 用户长期兴趣 embedding，[batch_size, output_dim]
        """
        b, n, d = x.size()
        # 在序列前添加 CLS token
        cls_tokens = self.cls_token.expand(b, -1, -1)  # [batch_size, 1, dims]
        x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, N+1, dims]
        # Transformer 编码
        x_transformed = self.transformer(x)  # [batch_size, N+1, dims]
        # 取 CLS 位置的输出作为序列表示
        cls_out = x_transformed[:, 0, :]  # [batch_size, dims]
        # 映射并归一化
        out = self.proj(cls_out)  # [batch_size, output_dim]
        out = self.norm(out)
        out = F.normalize(out, p=2, dim=-1)  # 提前用L2归一化，方便后续计算余弦相似度
        return out

