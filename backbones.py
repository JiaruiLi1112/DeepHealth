import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Args:
        n_embd (int): Embedding dimension.
        n_head (int): Number of attention heads.
        attn_pdrop (float): Attention dropout probability.
    """

    def __init__(
            self,
            n_embd: int,
            n_head: int,
            attn_pdrop: float = 0.1,
    ):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.o_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_pdrop = attn_pdrop

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,  # (B, L, L)
    ) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv_proj(x)  # (B, L, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape_heads(t):
            # (B, H, L, d)
            return t.view(B, L, self.n_head, self.head_dim).transpose(1, 2)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_pdrop,
        )  # (B, H, L, d)

        attn = attn.transpose(1, 2).contiguous().view(B, L, D)  # (B, L, D)
        return self.o_proj(attn)


class Block(nn.Module):
    """
    Transformer block consisting of self-attention and MLP.

    Args:
        n_embd (int): Embedding dimension.
        n_head (int): Number of attention heads.
        pdrop (float): Dropout probability.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        pdrop: float = 0.0,
    ):
        super().__init__()
        attn_pdrop = pdrop

        self.norm_1 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
        )
        self.norm_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(n_embd, 4 * n_embd),
            c_proj=nn.Linear(4 * n_embd, n_embd),
            act=nn.GELU(),
            dropout=nn.Dropout(pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(
            m.c_proj(m.act(m.c_fc(x))))
        self.resid_dropout = nn.Dropout(pdrop)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention
        h = self.norm_1(x)
        h = self.attn(h, attn_mask=attn_mask)
        x = x + self.resid_dropout(h)

        # MLP
        h = self.norm_2(x)
        h = self.mlpf(h)
        x = x + self.resid_dropout(h)

        return x
