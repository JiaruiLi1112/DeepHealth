import torch
import torch.nn as nn


class AgeSinusoidalEncoder(nn.Module):
    """
    Sinusoidal encoder for age.

    Args:
        n_embd (int): Embedding dimension. Must be even.
    """

    def __init__(self, n_embd: int):
        super().__init__()
        if n_embd % 2 != 0:
            raise ValueError("n_embd must be even for sinusoidal encoding.")
        self.n_embd = n_embd
        i = torch.arange(0, self.n_embd, 2, dtype=torch.float32)
        divisor = torch.pow(10000, i / self.n_embd)
        self.register_buffer('divisor', divisor)

    def forward(self, ages: torch.Tensor) -> torch.Tensor:
        t_years = ages / 365.25
        # Broadcast (B, L, 1) against (1, 1, D/2) to get (B, L, D/2)
        args = t_years.unsqueeze(-1) / self.divisor.view(1, 1, -1)
        # Interleave cos and sin along the last dimension
        output = torch.zeros(
            ages.shape[0], ages.shape[1], self.n_embd, device=ages.device)
        output[:, :, 0::2] = torch.cos(args)
        output[:, :, 1::2] = torch.sin(args)
        return output


class AgeMLPEncoder(nn.Module):
    """
    MLP encoder for age, using sinusoidal encoding as input.

    Args:
        n_embd (int): Embedding dimension.
    """

    def __init__(self, n_embd: int):
        super().__init__()

        self.sin_encoder = AgeSinusoidalEncoder(n_embd=n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, ages: torch.Tensor) -> torch.Tensor:
        x = self.sin_encoder(ages)
        output = self.mlp(x)
        return output
