import torch
import torch.nn as nn
import torch.nn.functional as F
from age_encoder import AgeSinusoidalEncoder, AgeMLPEncoder
from backbones import Block
from typing import Optional, List


class TabularEncoder(nn.Module):
    """
    Encoder for tabular features (continuous and categorical).

    Args:
        n_embd (int): Embedding dimension.
        n_cont (int): Number of continuous features.
        n_cate (int): Number of categorical features.
        cate_dims (List[int]): List of dimensions for each categorical feature.
    """

    def __init__(
            self,
            n_embd: int,
            n_cont: int,
            n_cate: int,
            cate_dims: List[int],
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_cont = n_cont
        self.n_cate = n_cate

        if n_cont > 0:
            hidden = 2 * n_embd
            self.cont_mlp = nn.Sequential(
                nn.Linear(2 * n_cont, hidden),
                nn.GELU(),
                nn.Linear(hidden, n_embd),
            )
        else:
            self.cont_mlp = None

        if n_cate > 0:
            assert len(cate_dims) == n_cate, \
                "Length of cate_dims must match n_cate"
            self.cate_embds = nn.ModuleList([
                nn.Embedding(dim, n_embd) for dim in cate_dims
            ])
            self.cate_mask_embds = nn.ModuleList([
                nn.Embedding(2, n_embd) for _ in range(n_cate)
            ])
        else:
            self.cate_embds = None
            self.cate_mask_embds = None

        self.cont_mask_proj = (
            nn.Linear(n_cont, n_embd) if n_cont > 0 else None
        )

        self.film = nn.Sequential(
            nn.Linear(n_embd, 2 * n_embd),
            nn.GELU(),
            nn.Linear(2 * n_embd, 2 * n_embd),
        )

        with torch.no_grad():
            for m in self.modules():
                last_linear = self.film[-1]
                if isinstance(m, nn.Linear):
                    last_linear.weight.zero_()
                    last_linear.bias.zero_()

    def forward(
        self,
        cont_features: Optional[torch.Tensor],
        cate_features: Optional[torch.Tensor],
    ) -> torch.Tensor:

        if self.n_cont == 0 and self.n_cate == 0:
            # infer (B, L) from whichever input is not None
            if cont_features is not None:
                B, L = cont_features.shape[:2]
                device = cont_features.device
            elif cate_features is not None:
                B, L = cate_features.shape[:2]
                device = cate_features.device
            else:
                raise ValueError(
                    "TabularEncoder received no features but cannot infer (B, L)."
                )
            return torch.zeros(B, L, self.n_embd, device=device)

        value_parts: List[torch.Tensor] = []
        mask_parts: List[torch.Tensor] = []

        if self.n_cont > 0 and cont_features is not None:
            if cont_features.dim() != 3:
                raise ValueError(
                    "cont_features must be 3D tensor (B, L, n_cont)")
            B, L, D_cont = cont_features.shape
            if D_cont != self.n_cont:
                raise ValueError(
                    f"Expected cont_features last dim to be {self.n_cont}, got {D_cont}")

            cont_mask = (~torch.isnan(cont_features)).float()
            cont_filled = torch.nan_to_num(cont_features, nan=0.0)
            cont_joint = torch.cat([cont_filled, cont_mask], dim=-1)
            h_cont_value = self.cont_mlp(cont_joint)
            value_parts.append(h_cont_value)

            if self.cont_mask_proj is not None:
                h_cont_mask = self.cont_mask_proj(cont_mask)
                mask_parts.append(h_cont_mask)

        if self.n_cate > 0 and cate_features is not None:
            if cate_features.dim() != 3:
                raise ValueError(
                    "cate_features must be 3D tensor (B, L, n_cate)")
            B, L, D_cate = cate_features.shape
            if D_cate != self.n_cate:
                raise ValueError(
                    f"Expected cate_features last dim to be {self.n_cate}, got {D_cate}")

            for i in range(self.n_cate):
                cate_feat = cate_features[:, :, i]
                cate_embd = self.cate_embds[i]
                cate_mask_embd = self.cate_mask_embds[i]

                cate_value = cate_embd(
                    torch.clamp(cate_feat, min=0))
                cate_mask = (cate_feat > 0).long()
                cate_mask_value = cate_mask_embd(cate_mask)

                value_parts.append(cate_value)
                mask_parts.append(cate_mask_value)

        h_value = torch.stack(value_parts, dim=0).sum(dim=0)
        h_mask = torch.stack(mask_parts, dim=0).sum(dim=0)
        h_mask_flat = h_mask.view(-1, self.n_embd)
        film_params = self.film(h_mask_flat)
        gamma_delta, beta = film_params.chunk(2, dim=-1)
        gamma = 1.0 + gamma_delta
        h_value_flat = h_value.view(-1, self.n_embd)
        h_out = gamma * h_value_flat + beta
        h_out = h_out.view(B, L, self.n_embd)
        return h_out


def _build_time_padding_mask(
    event_seq: torch.Tensor,
    time_seq: torch.Tensor,
) -> torch.Tensor:
    t_i = time_seq.unsqueeze(-1)
    t_j = time_seq.unsqueeze(1)
    time_mask = (t_j <= t_i)  # allow attending only to past or current
    key_is_valid = (event_seq != 0)  # disallow padded positions
    allowed = time_mask & key_is_valid.unsqueeze(1)
    attn_mask = ~allowed  # True means mask for scaled_dot_product_attention
    return attn_mask.unsqueeze(1)  # (B, 1, L, L)


class DelphiFork(nn.Module):
    """
    DelphiFork model for time-to-event prediction.

    Args:
        n_disease (int): Number of disease tokens.
        n_tech_tokens (int): Number of technical tokens.
        n_embd (int): Embedding dimension.
        n_head (int): Number of attention heads.
        n_layer (int): Number of transformer layers.
        n_cont (int): Number of continuous features.
        n_cate (int): Number of categorical features.
        cate_dims (List[int]): List of dimensions for each categorical feature.
        age_encoder_type (str): Type of age encoder ("sinusoidal" or "mlp").
        pdrop (float): Dropout probability.
        token_pdrop (float): Token dropout probability.
        n_dim (int): Dimension of theta parameters.
    """

    def __init__(
            self,
            n_disease: int,
            n_tech_tokens: int,
            n_embd: int,
            n_head: int,
            n_layer: int,
            n_cont: int,
            n_cate: int,
            cate_dims: List[int],
            age_encoder_type: str = "sinusoidal",
            pdrop: float = 0.0,
            token_pdrop: float = 0.0,
            n_dim: int = 1,
    ):
        super().__init__()
        self.vocab_size = n_disease + n_tech_tokens
        self.n_tech_tokens = n_tech_tokens
        self.n_disease = n_disease
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_dim = n_dim

        self.token_embedding = nn.Embedding(
            self.vocab_size, n_embd, padding_idx=0)
        if age_encoder_type == "sinusoidal":
            self.age_encoder = AgeSinusoidalEncoder(n_embd)
        elif age_encoder_type == "mlp":
            self.age_encoder = AgeMLPEncoder(n_embd)
        else:
            raise ValueError(
                f"Unsupported age_encoder_type: {age_encoder_type}")
        self.sex_encoder = nn.Embedding(2, n_embd)
        self.tabular_encoder = TabularEncoder(
            n_embd, n_cont, n_cate, cate_dims)

        self.blocks = nn.ModuleList([
            Block(
                n_embd=n_embd,
                n_head=n_head,
                pdrop=pdrop,
            ) for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.token_dropout = nn.Dropout(token_pdrop)

        # Head layers
        self.theta_proj = nn.Linear(n_embd, n_disease * n_dim)

    def forward(
            self,
            event_seq: torch.Tensor,  # (B, L)
            time_seq: torch.Tensor,  # (B, L)
            sex: torch.Tensor,  # (B,)
            cont_seq: torch.Tensor,  # (B, Lc, n_cont)
            cate_seq: torch.Tensor,  # (B, Lc, n_cate)
            b_prev: Optional[torch.Tensor] = None,  # (M,)
            t_prev: Optional[torch.Tensor] = None,  # (M,)
    ) -> torch.Tensor:
        token_embds = self.token_embedding(event_seq)  # (B, L, D)
        age_embds = self.age_encoder(time_seq)  # (B, L, D)
        sex_embds = self.sex_encoder(sex.unsqueeze(-1))  # (B, 1, D)
        table_embds = self.tabular_encoder(cont_seq, cate_seq)  # (B, Lc, D)
        mask = (event_seq == 1)  # (B, L)
        B, L = event_seq.shape
        Lc = table_embds.size(1)
        D = table_embds.size(2)

        # occ[b, t] = 第几次出现(从0开始)；非mask位置值无意义，后面会置0
        # (B, L), DOA: 0,1,2,...
        occ = torch.cumsum(mask.to(torch.long), dim=1) - 1

        # 将超过 Lc-1 的部分截断；并把非mask位置强制为 0（避免无意义 gather）
        tab_idx = occ.clamp(min=0, max=max(Lc - 1, 0))
        tab_idx = tab_idx.masked_fill(~mask, 0)  # (B, L)

        # 按 dim=1 从 (B, Lc, D) 取出每个位置应注入的 tab embedding -> (B, L, D)
        tab_inject = table_embds.gather(
            dim=1,
            index=tab_idx.unsqueeze(-1).expand(-1, -1, D)
        )
        # 只在 mask==True 的位置替换
        final_embds = torch.where(mask.unsqueeze(-1), tab_inject, token_embds)

        x = final_embds + age_embds + sex_embds  # (B, L, D)
        x = self.token_dropout(x)
        attn_mask = _build_time_padding_mask(
            event_seq, time_seq)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        x = self.ln_f(x)

        if b_prev is not None and t_prev is not None:
            M = b_prev.numel()
            c = x[b_prev, t_prev]  # (M, D)

            theta = self.theta_proj(c)  # (M, N_disease * n_dim)
            theta = theta.view(M, self.n_disease, self.n_dim)
            return theta
        else:
            return x
