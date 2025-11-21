import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List
from layers.TS_Pos_Enc import get_activation_fn


class Expert(nn.Module):
    def __init__(self, d_model, d_ff, activation="gelu", dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)

class MOEFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256, num_shared_experts=1, num_task_experts=2,
                 num_tasks=2, activation='gelu', dropout=0., self_exp_res_connect=True):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_shared_experts = num_shared_experts
        self.num_task_experts = num_task_experts
        self.self_exp_res_connect = self_exp_res_connect

        self.total_experts = num_tasks * num_task_experts + num_shared_experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, activation, dropout) for _ in range(self.total_experts)
        ])

        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, self.total_experts),
                nn.Softmax(dim=-1)
            ) for _ in range(num_tasks)
        ])

        if self_exp_res_connect:
            self.self_exp_weights = nn.ParameterList([
                nn.Parameter(torch.ones(self.total_experts)) for _ in range(num_tasks)
            ])

    def forward(self, inputs: Tensor) -> Tensor:
        """
        inputs: [B, T, D] 其中 T=num_tasks=2 (enc_out, embeddings)
        return: [B, T, D]
        """
        B, T, D = inputs.shape
        assert T == self.num_tasks

        experts_out = torch.stack([
            expert(inputs.reshape(-1, D)).reshape(B, T, -1)
            for expert in self.experts
        ], dim=2)  # [B, T, E, D]

        fused_out = []
        for t in range(self.num_tasks):
            gate_w = self.gates[t](inputs[:, t, :])  # [B, E]
            if self.self_exp_res_connect:
                gate_w = gate_w + self.self_exp_weights[t]
            gate_w = F.softmax(gate_w, dim=-1)  # 归一化
            out_t = torch.einsum("be, bted -> btd", gate_w, experts_out)[:, t, :]  # [B, D]
            fused_out.append(out_t)

        fused_out = torch.stack(fused_out, dim=1)  # [B, T, D]
        return fused_out


class CrossModal(nn.Module):
    def __init__(self, d_model, d_ff=256, dropout=0., activation='gelu',
                 num_shared_experts=1, num_task_experts=2, n_heads=5, d_k=None, d_v=None,
                 norm='LayerNorm', attn_dropout=0, 
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False,):
        super().__init__()
        self.fusion = MOEFeedForward(
            d_model, d_ff, num_shared_experts=num_shared_experts,
            num_task_experts=num_task_experts, num_tasks=2,
            activation=activation, dropout=dropout
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None):

        B, N, C = q.shape
        _, M, _ = k.shape

        if N != M:
            attn = torch.bmm(q, k.transpose(1, 2))  # [B, N, M]
            attn = F.softmax(attn, dim=-1)
            k = torch.bmm(attn, k)  # [B, N, C]

        inputs = torch.stack([q, k], dim=1)  # [B, T=2, N, C]
        inputs = inputs.permute(0, 2, 1, 3).reshape(B * N, 2, C)  # [B*N, 2, C]

        out = self.fusion(inputs)  
        out = out[:, 0, :].reshape(B, N, C)  

        out = q + self.dropout(out)
        out = self.norm(out)
        return out

