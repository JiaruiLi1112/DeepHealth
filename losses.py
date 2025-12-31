import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Pair extraction (utility; not used by the losses below)
# ============================================================
def get_valid_pairs_and_dt(
    event_seqs: torch.Tensor,
    time_seqs: torch.Tensor,
    n_tech_tokens: int
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Extract valid event pairs (prev -> next) and compute dt in years.

    Args:
        event_seqs (torch.Tensor): Event sequences.
        time_seqs (torch.Tensor): Time sequences.
        n_tech_tokens (int): Number of technical tokens.

    Returns:
        Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        (dt, b_prev, t_prev, b_next, t_next) if valid pairs exist, else None.

    Notes:
        - Assumes strict right-padding.
        - Filters to next events that are disease tokens: token_id >= n_tech_tokens.
        - Filters to strictly positive dt.
    """
    real_mask = event_seqs >= 1
    idx = real_mask.nonzero(as_tuple=False)

    if idx.size(0) <= 1:
        return None

    same_batch = idx[1:, 0] == idx[:-1, 0]
    if not same_batch.any():
        return None

    prev_idx = idx[:-1][same_batch]
    next_idx = idx[1:][same_batch]

    b_next, t_next = next_idx[:, 0], next_idx[:, 1]
    valid_target = event_seqs[b_next, t_next] >= n_tech_tokens
    if not valid_target.any():
        return None

    prev_idx = prev_idx[valid_target]
    next_idx = next_idx[valid_target]

    b_prev, t_prev = prev_idx[:, 0], prev_idx[:, 1]
    b_next, t_next = next_idx[:, 0], next_idx[:, 1]

    dt = (time_seqs[b_next, t_next] -
          time_seqs[b_prev, t_prev]).to(torch.float32) / 365.25
    valid_dt = dt > 0
    if not valid_dt.any():
        return None

    dt = dt[valid_dt]
    b_prev = b_prev[valid_dt]
    t_prev = t_prev[valid_dt]
    b_next = b_next[valid_dt]
    t_next = t_next[valid_dt]

    return dt, b_prev, t_prev, b_next, t_next


# ============================================================
# Losses (clean interface): loss_fn(preds, target_events, dt) -> (nll, regularization)
# ============================================================
class ExponentialNLLLoss(nn.Module):
    """
    Competing risks exponential likelihood.

    The negative log-likelihood is given by:

    .. math::
        \\text{nll} = -\\log \\lambda_{k^*} + \\left(\\sum_k \\lambda_k\\right) \\cdot dt

    Args:
        eps (float): Small epsilon for numerical stability.
    """

    def __init__(
            self,
            lambda_reg: float = 0.0,
            eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.lambda_reg = lambda_reg

    def forward(
        self,
        logits: torch.Tensor,
        target_events: torch.Tensor,
        dt: torch.Tensor,
        reduction: str = "mean",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            logits (torch.Tensor): (M, K) tensor of logits.
            target_events (torch.Tensor): (M,) int64 tensor of target events in [0, K).
            dt (torch.Tensor): (M,) float tensor of time intervals (years), strictly positive.
            reduction (str): 'mean', 'sum', or 'none'.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (nll, regularization) where regularization is 0.
        """
        logits = logits.squeeze(-1) if logits.dim() == 3 else logits
        hazards = F.softplus(logits) + self.eps  # (M,K)
        hazard_event = hazards.gather(
            1, target_events.unsqueeze(1)).squeeze(1)  # (M,)
        total_hazard = hazards.sum(dim=1)  # (M,)
        log_hazards = torch.log(hazards)               # (M, K)
        nll = -torch.log(hazard_event) + total_hazard * dt

        if reduction == "mean":
            nll = nll.mean()
        elif reduction == "sum":
            nll = nll.sum()

        reg = F.cross_entropy(log_hazards, target_events,
                              reduction="mean") * self.lambda_reg
        return nll, reg


class WeibullNLLLoss(nn.Module):
    """
    Weibull hazard in t.

    .. math::
        \\Lambda_k(t) = \\text{scale}_k \\cdot t^{\\text{shape}_k}

        \\lambda_k(t) = \\text{shape}_k \\cdot \\text{scale}_k \\cdot t^{\\text{shape}_k-1}

    Args:
        eps (float): Small epsilon for numerical stability.
        lambda_reg (float): Regularization weight.
        use_interval_near_integer (bool): If True, use interval likelihood for near-integer-year samples.
        near_integer_eps_years (float): Near-integer threshold in years.
        interval_half_width_years (float): Half-width \u0394 for interval [t-\u0394, t+\u0394] in years.
        min_integer_year (float): Only apply near-integer logic when round(t) >= min_integer_year.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        lambda_reg: float = 0.0,
    ):
        super().__init__()
        self.eps = eps
        self.lambda_reg = lambda_reg

    def _cif_target_weibull(
        self,
        scales: torch.Tensor,
        shapes: torch.Tensor,
        t: torch.Tensor,
        target_events: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute CIF_k(t) for the target cause via 16-pt Gauss-Legendre quadrature.

        CIF_k(t) = \int_0^t \lambda_k(u) S(u) du, with S(u)=exp(-\Lambda_total(u)).
        Interval probability mass satisfies P(a<=T<=b,K=k) = CIF_k(b) - CIF_k(a).
        """
        eps = self.eps
        t = torch.clamp(t, min=eps)

        x_nodes, w = _gauss_legendre_16(device=t.device, dtype=t.dtype)

        # Map nodes from [-1,1] -> [0,t]: u = 0.5*t*(x+1)
        u = 0.5 * t.unsqueeze(1) * (x_nodes.unsqueeze(0) + 1.0)  # (N, 16)
        u = torch.clamp(u, min=eps)
        log_u = torch.log(u)  # (N, 16)

        # \Lambda_total(u) = sum_k scale_k * u^{shape_k}
        # Compute u^{shape} in a dtype-friendly way via exp(shape * log u).
        u_pow_shape = torch.exp(log_u.unsqueeze(
            2) * shapes.unsqueeze(1))  # (N, 16, K)
        Lambda_total = torch.sum(scales.unsqueeze(
            1) * u_pow_shape, dim=2)  # (N, 16)
        survival = torch.exp(-Lambda_total)  # (N, 16)

        # Target hazard \lambda_{k*}(u) = shape*scale*u^{shape-1}
        shape_event = shapes.gather(
            1, target_events.unsqueeze(1)).squeeze(1)  # (N,)
        scale_event = scales.gather(
            1, target_events.unsqueeze(1)).squeeze(1)  # (N,)
        u_pow_shape_minus1 = torch.exp(
            (shape_event.unsqueeze(1) - 1.0) * log_u)  # (N, 16)
        hazard_event_u = shape_event.unsqueeze(
            1) * scale_event.unsqueeze(1) * u_pow_shape_minus1  # (N, 16)

        # Integral: 0.5*t*sum_q w_q * hazard_event(u_q) * S(u_q)
        integrand = hazard_event_u * survival
        cif = 0.5 * t * torch.sum(w.unsqueeze(0) * integrand, dim=1)  # (N,)

        # Safety clamp: CIF is a probability in [0,1] up to numerical error.
        cif = torch.clamp(cif, min=0.0, max=1.0)
        return cif

    def forward(
        self,
        logits: torch.Tensor,
        target_events: torch.Tensor,
        dt: torch.Tensor,
        reduction: str = "mean",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            logits (torch.Tensor): (M, K, 2) tensor of logits.
            target_events (torch.Tensor): (M,) tensor of target events.
            dt (torch.Tensor): (M,) tensor of time intervals.
            reduction (str): 'mean', 'sum', or 'none'.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (nll, regularization).
        """
        shapes = F.softplus(logits[..., 0]) + self.eps  # (M,K)
        scales = F.softplus(logits[..., 1]) + self.eps  # (M,K)
        eps = self.eps
        t = torch.clamp(dt, min=eps)

        t_mat = t.unsqueeze(1)  # (M,1)

        # cumulative hazard (M,K)
        cum_hazard = scales * t_mat.pow(shapes)

        # hazard (M,K)
        hazard = shapes * scales * t_mat.pow(shapes - 1.0)

        hazard_event = hazard.gather(1, target_events.unsqueeze(1)).squeeze(1)
        # Point-event likelihood: f_k(t) = \lambda_k(t) * exp(-\Lambda_total(t))
        # NLL_point = -log \lambda_{k*}(t) + \Lambda_total(t)
        nll = -torch.log(hazard_event + eps) + cum_hazard.sum(dim=1)

        if reduction == "mean":
            nll = nll.mean()
        elif reduction == "sum":
            nll = nll.sum()

        reg = shapes.new_zeros(())
        if self.lambda_reg > 0:
            reg = self.lambda_reg * (
                (torch.log(scales + eps) ** 2).mean() +
                (torch.log(shapes + eps) ** 2).mean()
            )
        return nll, reg


def _gauss_legendre_16(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    # Nodes and weights for 16-point Gauss-Legendre quadrature on [-1, 1]
    x = torch.tensor(
        [
            -0.989400934991649932596154173450,
            -0.944575023073232576077988415535,
            -0.865631202387831743880467897712,
            -0.755404408355003033895101194847,
            -0.617876244402643748446671764049,
            -0.458016777657227386342419442984,
            -0.281603550779258913230460501460,
            -0.095012509837637440185319335425,
            0.095012509837637440185319335425,
            0.281603550779258913230460501460,
            0.458016777657227386342419442984,
            0.617876244402643748446671764049,
            0.755404408355003033895101194847,
            0.865631202387831743880467897712,
            0.944575023073232576077988415535,
            0.989400934991649932596154173450,
        ],
        device=device,
        dtype=dtype,
    )
    w = torch.tensor(
        [
            0.027152459411754094851780572456,
            0.062253523938647892862843836994,
            0.095158511682492784809925107602,
            0.124628971255533872052476282192,
            0.149595988816576732081501730547,
            0.169156519395002538189312079030,
            0.182603415044923588866763667969,
            0.189450610455068496285396723208,
            0.189450610455068496285396723208,
            0.182603415044923588866763667969,
            0.169156519395002538189312079030,
            0.149595988816576732081501730547,
            0.124628971255533872052476282192,
            0.095158511682492784809925107602,
            0.062253523938647892862843836994,
            0.027152459411754094851780572456,
        ],
        device=device,
        dtype=dtype,
    )
    return x, w


class LogNormalBasisHazardLoss(nn.Module):
    """
    Competing risks hazard loss with RBF kernels in log-time.

    .. math::
        \\log \\lambda_k(t|x) = \\sum_{b=1}^{B} c_{k,b}(x) \\cdot \\exp\\left( - \\frac{(\\log t - \\mu_b)^2}{2 \\sigma^2} \\right)

    where :math:`\\mu_b` are fixed centers (log-time) and :math:`\\sigma` is a bandwidth parameter.
    The cumulative hazard is computed via numerical integration.

    Args:
        centers (Sequence[float]): Fixed centers in time domain (will be logged).
        bandwidth_scale (float): Scaling factor for initial bandwidth (relative to median center spacing).
        eps (float): Small epsilon.
        lambda_reg (float): Regularization weight.
        use_interval_near_integer (bool): If True, use interval likelihood for near-integer-year samples.
        near_integer_eps_years (float): Near-integer threshold in years.
        interval_half_width_years (float): Half-width \u0394 for interval [t-\u0394, t+\u0394] in years.
        min_integer_year (float): Only apply near-integer logic when round(t) >= min_integer_year.
    """

    def __init__(
        self,
        centers: Sequence[float],
        bandwidth_scale: float = 1.0,
        eps: float = 1e-6,
        lambda_reg: float = 0.0,
    ):
        super().__init__()
        self.eps = eps
        self.lambda_reg = lambda_reg
        self.num_basis = len(centers)

        # Fixed centers in log-time
        # centers should be strictly positive
        mu_list = [math.log(c) for c in centers]
        mu = torch.tensor(mu_list, dtype=torch.float32)
        self.register_buffer("mu", mu)

        # Heuristic for bandwidth: proportional to median spacing
        if self.num_basis > 1:
            sorted_mu, _ = torch.sort(mu)
            diffs = sorted_mu[1:] - sorted_mu[:-1]
            median_dist = torch.median(diffs).item()
            init_sigma = bandwidth_scale * median_dist
        else:
            init_sigma = bandwidth_scale  # Fallback

        # Parameterize as log_sigma to ensure positivity
        self.log_sigma = nn.Parameter(torch.tensor(math.log(init_sigma)))

    def _compute_kernel(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute RBF kernels on log-time.
        K_b(t) = exp( - (log t - mu_b)^2 / (2 sigma^2) )

        Args:
            t: (N,) tensor of times

        Returns:
            (N, B) tensor of kernel values
        """
        sigma = torch.exp(self.log_sigma)
        t_clamped = torch.clamp(t, min=self.eps)
        log_t = torch.log(t_clamped)  # (N,)

        # Expand for broadcasting
        # log_t: (N, 1)
        # mu: (1, B)
        diff = log_t.unsqueeze(1) - self.mu.unsqueeze(0)  # (N, B)

        # RBF kernel
        kernel_val = torch.exp(-(diff ** 2) / (2 * sigma ** 2))

        return kernel_val

    def forward(
        self,
        coeffs: torch.Tensor,        # (M, K, B)
        target_events: torch.Tensor,  # (M,)
        dt: torch.Tensor,            # (M,)
        reduction: str = "mean",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            coeffs (torch.Tensor): (M, K, B) tensor of coefficients for log-hazard.
            target_events (torch.Tensor): (M,) tensor of target events.
            dt (torch.Tensor): (M,) tensor of time intervals.
            reduction (str): 'mean' or 'none'.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (nll, regularization).
        """
        eps = self.eps
        t = torch.clamp(dt, min=eps)

        # 1. Event term: -log lambda_{k*}(t)
        # log lambda_k(t) = sum_b c_{kb} K_b(t)
        K_t = self._compute_kernel(t)  # (M, B)

        # log_hazards: (M, K)
        log_hazards = torch.einsum("mkb,mb->mk", coeffs, K_t)

        # Clamp log-hazards for numerical stability
        log_hazards = torch.clamp(log_hazards, max=20.0)

        # Select k*: (M,)
        log_hazard_event = log_hazards.gather(
            1, target_events.unsqueeze(1)).squeeze(1)

        # 2. Integral term: int_0^t sum_k lambda_k(s) ds
        # Quadrature
        x_nodes, w = _gauss_legendre_16(
            device=t.device, dtype=t.dtype)  # (16,), (16,)

        # Map nodes to [0, t]: s = t/2 * (x + 1)
        # s: (M, 16)
        s = 0.5 * t.unsqueeze(1) * (x_nodes.unsqueeze(0) + 1.0)

        # Kernels at quadrature points: (M*16, B) -> (M, 16, B)
        K_s = self._compute_kernel(s.reshape(-1))
        K_s = K_s.view(s.shape[0], s.shape[1], -1)

        # log lambda_k(s): (M, K, 16)
        # coeffs: (M, K, B)
        # K_s: (M, 16, B)
        # einsum: mkb, mqb -> mkq
        log_hazards_s = torch.einsum("mkb,mqb->mkq", coeffs, K_s)
        log_hazards_s = torch.clamp(log_hazards_s, max=20.0)

        # sum_k lambda_k(s) = sum_k exp(log_lambda_k(s))
        # Use logsumexp for stability: log(sum_k lambda_k) = logsumexp(log_hazards_s, dim=1)
        # Then exp to get sum_k lambda_k
        total_hazard_s = torch.logsumexp(log_hazards_s, dim=1).exp()  # (M, 16)

        # Integral: t/2 * sum_q w_q * total_hazard_s_q
        # w: (16,)
        integral = 0.5 * t * torch.sum(w.unsqueeze(0) * total_hazard_s, dim=1)

        nll = -log_hazard_event + integral

        # Regularization
        reg = coeffs.new_zeros(())
        if self.lambda_reg > 0:
            # Penalize sigma (prevent 0 or inf)
            reg = reg + self.lambda_reg * (self.log_sigma ** 2).mean()

        if reduction == "mean":
            nll = nll.mean()
        elif reduction == "sum":
            nll = nll.sum()

        return nll, reg

    def predict_log_hazard(self, coeffs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict log-hazards for given times.

        Args:
            coeffs: (M, K, B)
            t: (M, H) or (M,) - Time points (must be positive)

        Returns:
            log_hazards: (M, H, K) or (M, K)
        """
        # Handle t shape
        if t.dim() == 1:
            t = t.unsqueeze(1)  # (M, 1)

        M, H = t.shape
        K = coeffs.shape[1]

        # Compute kernels: (M*H, B)
        K_t = self._compute_kernel(t.flatten())
        K_t = K_t.view(M, H, -1)  # (M, H, B)

        # log_hazards: (M, K, B) * (M, H, B) -> (M, H, K)
        # einsum: mkb, mhb -> mhk
        log_hazards = torch.einsum("mkb,mhb->mhk", coeffs, K_t)
        log_hazards = torch.clamp(log_hazards, max=20.0)

        return log_hazards

    def predict_total_cum_hazard(self, coeffs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict total cumulative hazard sum_k Lambda_k(t).

        Args:
            coeffs: (M, K, B)
            t: (M, H)

        Returns:
            total_cum_hazard: (M, H)
        """
        M, H = t.shape
        device = t.device
        dtype = t.dtype

        # Quadrature nodes
        x_nodes, w = _gauss_legendre_16(device, dtype)  # (16,), (16,)

        # Map nodes to [0, t]: s = t/2 * (x + 1)
        # s: (M, H, 16)
        s = 0.5 * t.unsqueeze(-1) * (x_nodes.view(1, 1, -1) + 1.0)

        # Kernels at quadrature points: (M, H, 16, B)
        K_s = self._compute_kernel(s.flatten()).view(M, H, 16, -1)

        # log lambda_k(s): (M, K, B) * (M, H, 16, B) -> (M, H, 16, K)
        # einsum: mkb, mhqb -> mhqk
        log_hazards_s = torch.einsum("mkb,mhqb->mhqk", coeffs, K_s)
        log_hazards_s = torch.clamp(log_hazards_s, max=20.0)

        # sum_k lambda_k(s)
        total_hazard_s = torch.logsumexp(
            log_hazards_s, dim=-1).exp()  # (M, H, 16)

        # Integral: t/2 * sum_q w_q * total_hazard_s_q
        # w: (16,)
        integral = 0.5 * t * \
            torch.sum(w.view(1, 1, -1) * total_hazard_s, dim=-1)

        return integral

    def predict_cif(self, coeffs: torch.Tensor, t_eval: torch.Tensor) -> torch.Tensor:
        """
        Predict CIF P(T <= t, K=k) using nested quadrature.

        Args:
            coeffs: (M, K, B)
            t_eval: (M, H)

        Returns:
            cif: (M, H, K)
        """
        M, H = t_eval.shape
        K = coeffs.shape[1]
        device = t_eval.device
        dtype = t_eval.dtype

        # Outer Quadrature (for CIF integral)
        x_nodes, w = _gauss_legendre_16(device, dtype)

        # u: (M, H, 16) points for outer integral
        u = 0.5 * t_eval.unsqueeze(-1) * (x_nodes.view(1, 1, -1) + 1.0)

        # 1. Compute lambda_k(u)
        # K_u: (M, H, 16, B)
        K_u = self._compute_kernel(u.flatten()).view(M, H, 16, -1)

        # log_hazards: (M, K, B) * (M, H, 16, B) -> (M, H, 16, K)
        # einsum: mkb, mhqb -> mhqk
        log_hazards = torch.einsum("mkb,mhqb->mhqk", coeffs, K_u)
        log_hazards = torch.clamp(log_hazards, max=20.0)
        hazards = torch.exp(log_hazards)  # (M, H, 16, K)

        # 2. Compute Lambda_total(u) via Inner Quadrature
        # Reuse predict_total_cum_hazard by flattening u
        # u_flat: (M, H*16)
        u_flat = u.view(M, -1)
        Lambda_total_u_flat = self.predict_total_cum_hazard(
            coeffs, u_flat)  # (M, H*16)
        Lambda_total_u = Lambda_total_u_flat.view(M, H, 16)

        # Survival S(u)
        survival = torch.exp(-Lambda_total_u)  # (M, H, 16)

        # Integrand for CIF: lambda_k(u) * S(u)
        # hazards: (M, H, 16, K)
        # survival: (M, H, 16)
        integrand = hazards * survival.unsqueeze(-1)  # (M, H, 16, K)

        # CIF Integral
        # w: (16,) -> (1, 1, 16, 1)
        # Multiply by (t/2) with correct broadcasting for batched M>1
        # t_eval: (M, H) -> (M, H, 1)
        cif = 0.5 * t_eval.unsqueeze(-1) * torch.sum(
            w.view(1, 1, -1, 1) * integrand, dim=2)  # (M, H, K)

        return cif
