import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from block import conv_1d

world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"


@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """

    dim: int = 211
    inter_dim: int = 256
    moe_inter_dim: int = 256
    n_routed_experts: int = 100
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_scale: float = 1.


class Expert(nn.Module):

    def __init__(self, dim, hidden_dim=64, dropout=0., residual=False):
        super().__init__()
        self.residual = residual
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.w3 = nn.Linear(dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x if self.residual is True else 0.
        x = self.norm(x)
        x = F.silu(self.w1(x)) * self.w3(x)
        return self.w2(self.dropout(x)) + y


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.
    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """

    def __init__(self, args: ModelArgs = ModelArgs()):
        """
        Initializes the Gate module.
        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.gate_weights = nn.Linear(args.dim, args.n_routed_experts)
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None
        # 添加专家使用频率统计
        self.register_buffer('expert_activation_counts', torch.zeros(args.n_routed_experts))
        self.register_buffer('expert_percentages', torch.ones(args.n_routed_experts))
        self.total_activations = 0
        self.use_load_balancing = True

    def update_expert_percentages(self):
        """更新专家使用百分比"""
        if self.total_activations > 0:
            # 计算每个专家的激活百分比
            self.expert_percentages = self.expert_activation_counts / self.total_activations
            # 为避免除零错误,确保没有零百分比
            self.expert_percentages = torch.clamp(self.expert_percentages, min=1e-5)

    def reset_counts(self):
        """重置专家计数,通常在每个epoch开始时调用"""
        self.expert_activation_counts.zero_()
        self.total_activations = 0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        # 计算初始路由分数
        scores = self.gate_weights(x)
        # 根据scoring函数选择不同的激活方式
        if self.score_func == "sigmoid":
            # 先计算激活值
            original_scores = scores.sigmoid()

            # 如果在训练模式且启用负载均衡,应用负载均衡修正
            if self.training and self.use_load_balancing:
                # 使用专家百分比进行负载均衡调整
                # 百分比越高的专家,其score越小
                balanced_scores = original_scores / self.expert_percentages.to(scores.device)
                scores = balanced_scores
            else:
                scores = original_scores
        else:  # "softmax"
            # 对于softmax,我们在计算exp之后应用负载均衡
            if self.training and self.use_load_balancing:
                # 计算exp值
                exp_scores = torch.exp(scores)
                # 应用负载均衡,除以专家激活百分比
                balanced_exp_scores = exp_scores / self.expert_percentages.to(scores.device).unsqueeze(0)
                # 重新归一化
                scores = balanced_exp_scores / balanced_exp_scores.sum(dim=-1, keepdim=True)
            else:
                scores = F.softmax(scores, dim=-1)
        # 保存原始分数用于返回权重
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        # 选出前k个专家
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        # 获取这些专家对应的权重
        weights = original_scores.gather(1, indices)
        # 如果使用sigmoid激活,需要重新归一化权重
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        # 在训练阶段统计专家激活次数
        if self.training:
            # 获取选中的每个专家的索引
            flat_indices = indices.flatten()
            # 统计每个专家的激活次数
            for expert_idx in range(self.expert_activation_counts.size(0)):
                self.expert_activation_counts[expert_idx] += (flat_indices == expert_idx).sum().item()
            # 更新总激活次数
            self.total_activations += flat_indices.size(0)
        return weights.type_as(x), indices


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """

    def __init__(self, args: ModelArgs = ModelArgs()):

        super().__init__()
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList(
            [Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
             for i in range(self.n_routed_experts)])
        self.shared_experts = Expert(args.dim, args.n_shared_experts * args.inter_dim)

    def update_expert_percentages(self):
        """更新专家使用比例"""
        self.gate.update_expert_percentages()

    def reset_counts(self):
        """重置专家激活计数"""
        self.gate.reset_counts()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        # counts 是一个列表，长度为 self.n_routed_experts，其中每个元素表示该专家被分配到的输入样本（token）数量
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)  # 每个expert只计算需要它的token
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        return (y + z).view(shape)
