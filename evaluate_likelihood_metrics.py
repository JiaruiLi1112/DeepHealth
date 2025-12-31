"""Evaluation script for likelihood-consistent survival metrics.

Implements 6 metrics:
1) Survival NLL (mean/median/p95)
2) Integrated Brier Score (IBS) (uncensored next-event data)
3) Horizon Brier scores
4) Calibration-in-the-large (CITL)
5) Calibration slope (logistic regression)
6) PIT diagnostics + KS uniformity statistic

This repo's model (`DelphiFork`) is a competing-risks model over next-event time
with parametric hazards (exponential/weibull) or a log-time RBF hazard basis
("lognormal" loss type).

The script evaluates on the *test split* created with the same ratios/seed as training.
Each valid (prev -> next) pair is treated as one evaluation instance with:
- t_event = dt in years
- event_type = next disease index in 1..K (0 would mean censored; not present in pair data)

Outputs:
- metrics_summary.json
- calibration_table_{H}.csv for each horizon
- pit_hist.csv

Usage example:
    python evaluate_likelihood_metrics.py --run_dir runs/weibull_sinusoidal_fullcov_20251218_095959

"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

from dataset import HealthDataset, health_collate_fn
from losses import (
    ExponentialNLLLoss,
    WeibullNLLLoss,
    LogNormalBasisHazardLoss,
    get_valid_pairs_and_dt,
    _gauss_legendre_16,
)
from model import DelphiFork


# -------------------------
# Numerics helpers
# -------------------------


def _progress(iterable, *, desc: str, total: Optional[int] = None, leave: bool = False):
    """Lightweight wrapper: uses tqdm if available, else returns iterable."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=leave)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def _ks_uniform(u: np.ndarray) -> Tuple[float, float]:
    """One-sample KS test vs Uniform(0,1).

    Returns (D, pvalue) using asymptotic Kolmogorov distribution approximation.
    """
    u = np.asarray(u, dtype=np.float64)
    u = u[np.isfinite(u)]
    n = u.size
    if n == 0:
        return float("nan"), float("nan")

    u = np.sort(np.clip(u, 0.0, 1.0))
    i = np.arange(1, n + 1, dtype=np.float64)
    d_plus = np.max(i / n - u)
    d_minus = np.max(u - (i - 1) / n)
    d = float(max(d_plus, d_minus))

    # Asymptotic p-value: 2 * sum_{k>=1} (-1)^{k-1} exp(-2 k^2 n d^2)
    # Works reasonably for moderate/large n.
    if not np.isfinite(d) or d <= 0:
        return d, 1.0

    # Use effective n adjustment often used in practice
    en = math.sqrt(n)
    lam = (en + 0.12 + 0.11 / en) * d

    # Compute series with truncation
    s = 0.0
    for k in range(1, 200):
        term = 2.0 * ((-1.0) ** (k - 1)) * \
            math.exp(-2.0 * (k * k) * (lam * lam))
        s += term
        if abs(term) < 1e-12:
            break
    p = float(np.clip(s, 0.0, 1.0))
    return d, p


def _map_times_to_indices(union_times: np.ndarray, query_times: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Map query_times into indices of union_times, with tolerance.

    Assumes every query_time is present in union_times (up to tol).
    """
    union_times = np.asarray(union_times, dtype=np.float64)
    query_times = np.asarray(query_times, dtype=np.float64)
    idx = np.searchsorted(union_times, query_times, side="left")
    idx = np.clip(idx, 0, union_times.size - 1)

    # Fix off-by-one when searchsorted lands on the next element
    left = np.clip(idx - 1, 0, union_times.size - 1)
    choose_left = np.abs(
        union_times[left] - query_times) < np.abs(union_times[idx] - query_times)
    idx = np.where(choose_left, left, idx)

    if np.max(np.abs(union_times[idx] - query_times)) > tol:
        raise ValueError(
            "Some query times were not found in union_times within tolerance.")
    return idx.astype(np.int64)


def _interp1d_batch(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Vectorized linear interpolation.

    Args:
        x: (T,) sorted increasing.
        y: (N,T)
        xq: (N,)

    Returns:
        yq: (N,)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xq = np.asarray(xq, dtype=np.float64)

    T = x.size
    if T < 2:
        return y[:, 0].copy()

    j = np.searchsorted(x, xq, side="right") - 1
    j = np.clip(j, 0, T - 2)

    x0 = x[j]
    x1 = x[j + 1]
    y0 = y[np.arange(y.shape[0]), j]
    y1 = y[np.arange(y.shape[0]), j + 1]

    denom = np.where(x1 > x0, (x1 - x0), 1.0)
    w = (xq - x0) / denom
    w = np.clip(w, 0.0, 1.0)
    return y0 + (y1 - y0) * w


# -------------------------
# Calibration regression (logistic)
# -------------------------

def compute_cal_slope_weighted_logistic(
    y: np.ndarray,
    p: np.ndarray,
    w: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> float:
    """Weighted logistic regression: logit(E[y]) = a + b * logit(p).

    Returns slope b.
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    x = _logit(p)
    X = np.column_stack([np.ones_like(x), x])  # (N,2)

    # Initialize close to identity calibration
    beta = np.array([0.0, 1.0], dtype=np.float64)

    for _ in range(max_iter):
        eta = X @ beta
        mu = _sigmoid(eta)
        mu = np.clip(mu, 1e-8, 1.0 - 1e-8)
        var = mu * (1.0 - mu)

        # IRLS working response
        z = eta + (y - mu) / var

        # Weighted least squares with weights: w * var
        ww = w * var
        XtW = X.T * ww  # (2,N)
        H = XtW @ X  # (2,2)
        rhs = XtW @ z  # (2,)

        # Ridge for stability
        ridge = 1e-8
        H_reg = H + ridge * np.eye(2)
        beta_new = np.linalg.solve(H_reg, rhs)

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    return float(beta[1])


# -------------------------
# Model prediction helpers
# -------------------------

@dataclass
class EvalConfig:
    run_dir: str
    checkpoint: str
    out_dir: str
    device: str
    batch_size: int
    t_max: float
    n_grid: int
    horizons: Tuple[float, ...]
    seed: int
    data_prefix: str
    loss_type: str
    full_cov: bool
    age_encoder: str
    n_embd: int
    n_layer: int
    n_head: int
    pdrop: float


def _load_train_config(run_dir: str) -> Dict:
    cfg_path = os.path.join(run_dir, "train_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_model_and_loss(cfg: Dict, dataset: HealthDataset, device: torch.device):
    loss_type = cfg["loss_type"]
    if loss_type == "exponential":
        criterion = ExponentialNLLLoss(lambda_reg=float(
            cfg.get("lambda_reg", 0.0))).to(device)
        n_dim = 1
    elif loss_type == "weibull":
        criterion = WeibullNLLLoss(lambda_reg=float(
            cfg.get("lambda_reg", 0.0))).to(device)
        n_dim = 2
    elif loss_type == "lognormal":
        # Centers must match the ones used during training.
        # NOTE: Training uses centers = list(train.bin_edges).
        from train import bin_edges  # local import to avoid circulars at module import time

        centers = list(bin_edges)
        criterion = LogNormalBasisHazardLoss(
            centers=centers,
            bandwidth_scale=1.0,
            lambda_reg=float(cfg.get("lambda_reg", 0.0)),
        ).to(device)
        n_dim = len(centers)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    model = DelphiFork(
        n_disease=dataset.n_disease,
        n_tech_tokens=2,
        n_cont=dataset.n_cont,
        n_cate=dataset.n_cate,
        cate_dims=dataset.cate_dims,
        n_embd=int(cfg["n_embd"]),
        n_layer=int(cfg["n_layer"]),
        n_head=int(cfg["n_head"]),
        pdrop=float(cfg.get("pdrop", 0.0)),
        age_encoder_type=str(cfg.get("age_encoder", "sinusoidal")),
        n_dim=int(n_dim),
    ).to(device)

    return model, criterion


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)


def _total_cumhaz_any(
    loss_type: str,
    logits: torch.Tensor,
    t: torch.Tensor,
    criterion: torch.nn.Module,
    eps: float = 1e-6,
    k_chunk: int = 128,
) -> torch.Tensor:
    r"""Compute total cumulative hazard \sum_k Lambda_k(t) for any-event distribution.

    Args:
        logits: (M,K,dim)
        t: (H,) times

    Returns:
        total_cumhaz: (M,H)
    """
    # IMPORTANT: Ensure Λ(0) = 0 exactly.
    # Some hazard parameterizations require strictly-positive t internally
    # (e.g., log(t) in Weibull/lognormal-basis). We therefore compute using a
    # safe t for those cases, but explicitly overwrite Λ(t==0) to zero.
    t = t.to(device=logits.device)
    zero_mask = t <= 0
    t_safe = torch.where(zero_mask, torch.as_tensor(
        eps, device=t.device, dtype=t.dtype), t)

    if loss_type == "exponential":
        if logits.dim() == 3:
            logits = logits.squeeze(-1)
        hazards = F.softplus(logits) + eps  # (M,K)
        total_hazard = hazards.sum(dim=1)  # (M,)
        # No need to clamp time here; multiplication preserves exact zero.
        total = total_hazard[:, None] * t[None, :]
        if torch.any(zero_mask):
            total = total.clone()
            total[:, zero_mask] = 0.0
        return total

    if loss_type == "weibull":
        shapes = F.softplus(logits[..., 0]) + eps  # (M,K)
        scales = F.softplus(logits[..., 1]) + eps  # (M,K)
        M, K = shapes.shape
        H = t_safe.numel()

        log_t = torch.log(t_safe).view(1, 1, H)  # (1,1,H)
        total = torch.zeros((M, H), device=logits.device, dtype=logits.dtype)

        for k0 in range(0, K, k_chunk):
            k1 = min(K, k0 + k_chunk)
            sh = shapes[:, k0:k1].unsqueeze(-1)  # (M,Kc,1)
            sc = scales[:, k0:k1].unsqueeze(-1)  # (M,Kc,1)
            # t^shape = exp(shape * log t)
            contrib = sc * torch.exp(sh * log_t)  # (M,Kc,H)
            total = total + contrib.sum(dim=1)

        if torch.any(zero_mask):
            total = total.clone()
            total[:, zero_mask] = 0.0
        return total

    if loss_type == "lognormal":
        # criterion methods expect t shaped (M,H)
        t_mat = t_safe.view(1, -1).expand(logits.shape[0], -1)
        total = criterion.predict_total_cum_hazard(logits, t_mat)
        if torch.any(zero_mask):
            total = total.clone()
            total[:, zero_mask] = 0.0
        return total

    raise ValueError(f"Unsupported loss_type: {loss_type}")


def _total_cumhaz_any_per_instance(
    loss_type: str,
    logits: torch.Tensor,
    t_event: torch.Tensor,
    criterion: torch.nn.Module,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Compute per-instance total cumulative hazard \sum_k Lambda_k(t_event[i]).

    Args:
        logits: (M,K,dim)
        t_event: (M,) times

    Returns:
        total_cumhaz: (M,)
    """
    # Ensure Λ(0)=0 exactly for per-instance evaluations (e.g., PIT at t_obs=0).
    t_event = t_event.to(device=logits.device)
    zero_mask = t_event <= 0
    t_safe = torch.where(zero_mask, torch.as_tensor(
        eps, device=t_event.device, dtype=t_event.dtype), t_event)

    if loss_type == "exponential":
        if logits.dim() == 3:
            logits = logits.squeeze(-1)
        hazards = F.softplus(logits) + eps  # (M,K)
        total_hazard = hazards.sum(dim=1)  # (M,)
        total = total_hazard * t_event
        if torch.any(zero_mask):
            total = total.clone()
            total[zero_mask] = 0.0
        return total

    if loss_type == "weibull":
        shapes = F.softplus(logits[..., 0]) + eps  # (M,K)
        scales = F.softplus(logits[..., 1]) + eps  # (M,K)
        t_mat = t_safe.unsqueeze(1)  # (M,1)
        cum_h = scales * t_mat.pow(shapes)
        total = cum_h.sum(dim=1)
        if torch.any(zero_mask):
            total = total.clone()
            total[zero_mask] = 0.0
        return total

    if loss_type == "lognormal":
        t_mat = t_safe.unsqueeze(1)  # (M,1)
        total = criterion.predict_total_cum_hazard(logits, t_mat).squeeze(1)
        if torch.any(zero_mask):
            total = total.clone()
            total[zero_mask] = 0.0
        return total

    raise ValueError(f"Unsupported loss_type: {loss_type}")


def _predict_cif_chunked(
    loss_type: str,
    logits: torch.Tensor,
    t_eval: torch.Tensor,
    criterion: torch.nn.Module,
    k_chunk: int,
    use_amp: bool = False,
    eps: float = 1e-6,
) -> Iterable[Tuple[int, int, torch.Tensor]]:
    """Yields (k0, k1, cif_chunk) where cif_chunk is (M, Kc, TU).

    Handles global survival computation internally to ensure correctness
    under competing risks, while chunking the disease dimension to avoid OOM.
    """
    t_eval = t_eval.to(device=logits.device)
    M, K = logits.shape[:2]
    TU = t_eval.numel()
    zero_mask = t_eval <= 0
    t_safe = torch.where(zero_mask, torch.as_tensor(
        eps, device=t_eval.device, dtype=t_eval.dtype), t_eval)

    # -------------------------------------------------------
    # 1. Compute Global Survival / Cumulative Hazard
    # -------------------------------------------------------
    # We need S(t) or S(u) depending on loss type.

    if loss_type == "exponential":
        # Simple: S(t) = exp(-sum_k lambda_k * t)
        # We can compute sum_k lambda_k globally.
        if logits.dim() == 3:
            logits_sq = logits.squeeze(-1)
        else:
            logits_sq = logits

        # Sum hazards over K
        total_h = torch.zeros((M,), device=logits.device, dtype=logits.dtype)
        for k0 in range(0, K, k_chunk):
            k1 = min(K, k0 + k_chunk)
            h_chunk = F.softplus(logits_sq[:, k0:k1]) + eps
            total_h += h_chunk.sum(dim=1)

        # S(t) = exp(-total_h * t)
        # We can compute this per chunk or globally. Globally is (M, TU).
        # If TU is huge, (M, TU) is fine (128*1000*4 bytes ~ 0.5MB).
        exp_term = torch.exp(-total_h[:, None] * t_eval[None, :])  # (M, TU)

        # Yield chunks
        for k0 in range(0, K, k_chunk):
            k1 = min(K, k0 + k_chunk)
            h_chunk = F.softplus(logits_sq[:, k0:k1]) + eps  # (M, Kc)

            # CIF_k(t) = (lambda_k / Lambda_total) * (1 - S(t))
            #          = (lambda_k / total_h) * (1 - exp(-total_h * t))
            frac = h_chunk / total_h[:, None].clamp_min(eps)  # (M, Kc)
            cif_chunk = frac[:, :, None] * \
                (1.0 - exp_term)[:, None, :]  # (M, Kc, TU)

            if torch.any(zero_mask):
                cif_chunk[:, :, zero_mask] = 0.0
            yield k0, k1, cif_chunk

    elif loss_type == "weibull":
        # S(t) = exp(-sum_k Lambda_k(t))
        # Lambda_k(t) = scale * t^shape
        # We compute sum_k Lambda_k(t) globally.

        total_cumhaz = torch.zeros(
            (M, TU), device=logits.device, dtype=logits.dtype)
        log_t = torch.log(t_safe).view(1, 1, TU)

        for k0 in range(0, K, k_chunk):
            k1 = min(K, k0 + k_chunk)
            l_chunk = logits[:, k0:k1, :]
            shapes = F.softplus(l_chunk[..., 0]) + eps
            scales = F.softplus(l_chunk[..., 1]) + eps

            # (M, Kc, TU)
            cumhaz_chunk = scales[:, :, None] * \
                torch.exp(shapes[:, :, None] * log_t)
            total_cumhaz += cumhaz_chunk.sum(dim=1)

        surv = torch.exp(-total_cumhaz)  # (M, TU)

        # S_avg for trapezoidal integration
        S_avg = 0.5 * (surv[:, 1:] + surv[:, :-1])  # (M, TU-1)

        for k0 in range(0, K, k_chunk):
            k1 = min(K, k0 + k_chunk)
            l_chunk = logits[:, k0:k1, :]
            shapes = F.softplus(l_chunk[..., 0]) + eps
            scales = F.softplus(l_chunk[..., 1]) + eps

            # Recompute cumhaz for chunk
            cumhaz_chunk = scales[:, :, None] * \
                torch.exp(shapes[:, :, None] * log_t)  # (M, Kc, TU)

            dLambda = cumhaz_chunk[:, :, 1:] - \
                cumhaz_chunk[:, :, :-1]  # (M, Kc, TU-1)
            integrand = dLambda * S_avg[:, None, :]  # (M, Kc, TU-1)

            cif_tail = torch.cumsum(integrand, dim=2)
            cif0 = logits.new_zeros((M, k1-k0, 1))
            cif_chunk = torch.cat([cif0, cif_tail], dim=2)  # (M, Kc, TU)

            if torch.any(zero_mask):
                cif_chunk[:, :, zero_mask] = 0.0
            yield k0, k1, cif_chunk

    elif loss_type == "lognormal":
        # Quadrature based integration
        # 1. Nodes
        x_nodes, w = _gauss_legendre_16(logits.device, logits.dtype)
        # u: (1, TU, 16) -> expand to (M, TU, 16)
        u = 0.5 * t_safe.view(1, -1, 1) * (x_nodes.view(1, 1, -1) + 1.0)
        u = u.expand(M, -1, -1).contiguous()
        u_flat = u.view(M, -1)  # (M, TU*16)

        # 2. Global Lambda_total at u
        Lambda_total_u_flat = torch.zeros_like(u_flat)

        for k0 in range(0, K, k_chunk):
            k1 = min(K, k0 + k_chunk)
            logits_chunk = logits[:, k0:k1, :]
            with torch.amp.autocast('cuda', enabled=use_amp):
                # predict_total_cum_hazard sums over K in the chunk
                chunk_cumhaz = criterion.predict_total_cum_hazard(
                    logits_chunk, u_flat)
            Lambda_total_u_flat += chunk_cumhaz

        S_u = torch.exp(-Lambda_total_u_flat).view(M, TU, 16)  # (M, TU, 16)

        # 3. CIF chunks
        for k0 in range(0, K, k_chunk):
            k1 = min(K, k0 + k_chunk)
            logits_chunk = logits[:, k0:k1, :]

            with torch.amp.autocast('cuda', enabled=use_amp):
                # log_hazards: (M, TU*16, Kc)
                log_hazards = criterion.predict_log_hazard(logits_chunk, u_flat)
                hazards = torch.exp(log_hazards).view(
                    M, TU, 16, k1-k0)  # (M, TU, 16, Kc)

                # Integrand: hazards * S_u
                integrand = hazards * S_u.unsqueeze(-1)  # (M, TU, 16, Kc)

                # Integrate: t/2 * sum(w * integrand)
                # w: (16,) -> (1, 1, 16, 1)
                # t_safe: (TU,) -> (1, TU, 1)
                cif_chunk = 0.5 * \
                    t_safe.view(1, -1, 1) * \
                    torch.sum(w.view(1, 1, -1, 1) * integrand, dim=2)
                # (M, TU, Kc) -> (M, Kc, TU)
                cif_chunk = cif_chunk.permute(0, 2, 1)

            if torch.any(zero_mask):
                cif_chunk[:, :, zero_mask] = 0.0
            yield k0, k1, cif_chunk

    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")


def _nll_per_instance(
    loss_type: str,
    logits: torch.Tensor,
    target_events_0based: torch.Tensor,
    t_event: torch.Tensor,
    is_censored: torch.Tensor,
    criterion: torch.nn.Module,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-instance NLL.

    If is_censored is True, returns -log S(t). Otherwise uses -log f_k(t).
    """
    t_event = torch.clamp(t_event, min=eps)
    if loss_type in ("exponential", "weibull", "lognormal"):
        # Use the criterion's reduction=none for event terms when available,
        # but add censoring handling explicitly.
        if loss_type == "lognormal":
            # LogNormalBasisHazardLoss returns -log lambda_k(t) + integral total hazard
            # which equals -log f_k(t) (since f_k = lambda_k * exp(-Lambda_total)).
            nll_event, _ = criterion(
                logits, target_events_0based, t_event, reduction="none")
        else:
            nll_event, _ = criterion(
                logits, target_events_0based, t_event, reduction="none")

        # Censoring term: -log S(t) = Lambda_total(t)
        # (works for all three variants)
        # For event cases, nll_event is already correct; for censored replace.
        if is_censored.any():
            total_cumhaz = _total_cumhaz_any_per_instance(
                loss_type=loss_type,
                logits=logits,
                t_event=t_event,
                criterion=criterion,
                eps=eps,
            )
            nll = torch.where(is_censored, total_cumhaz, nll_event)
            return nll

        return nll_event

    raise ValueError(f"Unsupported loss_type: {loss_type}")


# -------------------------
# Evaluation pipeline
# -------------------------

def _iter_eval_pairs(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield (dt_years, event_type_1based, logits) for each batch of extracted pairs."""
    model.eval()
    with torch.no_grad():
        total_batches: Optional[int]
        try:
            total_batches = len(loader)
        except Exception:
            total_batches = None

        for batch in _progress(loader, desc="Extract eval pairs", total=total_batches, leave=False):
            event_seq, time_seq, cont_feats, cate_feats, sexes = batch
            event_seq = event_seq.to(device)
            time_seq = time_seq.to(device)
            cont_feats = cont_feats.to(device)
            cate_feats = cate_feats.to(device)
            sexes = sexes.to(device)

            res = get_valid_pairs_and_dt(event_seq, time_seq, 2)
            if res is None:
                continue
            dt, b_prev, t_prev, b_next, t_next = res
            logits = model(
                event_seq,
                time_seq,
                sexes,
                cont_feats,
                cate_feats,
                b_prev=b_prev,
                t_prev=t_prev,
            )
            target_events_0 = (
                event_seq[b_next, t_next] - 2).to(torch.long)  # 0..K-1
            event_type_1 = target_events_0 + 1
            yield dt, event_type_1, logits


def compute_metrics(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    loader: DataLoader,
    loss_type: str,
    t_grid: np.ndarray,
    horizons: Sequence[float],
    out_dir: str,
    pit_bins: int = 20,
    seed: int = 0,
    k_chunk: int = 64,
    t_chunk: int = 0,
    use_amp_eval: bool = False,
    empty_cache_debug: bool = False,
) -> Dict[str, float]:
    device = next(model.parameters()).device
    print("[1/5] Computing metrics: iterating batches, NLL/Brier/PIT...")
    # Treat all evaluation pairs as observed next-event data (no censoring).
    # Metrics are computed unweighted.
    n_total = 0

    nll_list: List[np.ndarray] = []
    pit_u: List[np.ndarray] = []

    # Horizon-level arrays for any-event calibration/slope
    p_h_list: List[np.ndarray] = [[] for _ in horizons]  # type: ignore
    y_h_list: List[np.ndarray] = [[] for _ in horizons]  # type: ignore
    w_h_list: List[np.ndarray] = [[] for _ in horizons]  # type: ignore

    # Disease-specific accumulators (initialized lazily when we see first batch)
    K: Optional[int] = None
    nll_by_k: Optional[List[List[np.ndarray]]] = None
    brier_sum_grid_by_k: Optional[np.ndarray] = None  # (K,G)
    brier_sum_h_by_k: Optional[np.ndarray] = None  # (K,H)
    # Calibration binning (streaming, fixed bins)
    cal_bins = 10
    cal_bin_count: Optional[np.ndarray] = None  # (K,H,B)
    cal_bin_sum_p: Optional[np.ndarray] = None  # (K,H,B)
    cal_bin_sum_y: Optional[np.ndarray] = None  # (K,H,B)
    citl_pred_sum: Optional[np.ndarray] = None  # (K,H)
    citl_obs_sum: Optional[np.ndarray] = None  # (K,H)
    logloss_sum: Optional[np.ndarray] = None  # (K,H)
    logloss_count: Optional[np.ndarray] = None  # (K,H)
    pit_u_by_k: Optional[List[List[np.ndarray]]] = None

    horizons_arr = np.asarray(horizons, dtype=np.float64)
    t_union = np.unique(np.concatenate(
        [t_grid.astype(np.float64), horizons_arr.astype(np.float64)], axis=0))
    if t_union.size == 0:
        raise ValueError("t_union empty")
    if t_union[0] > 0.0:
        t_union = np.concatenate(
            [np.array([0.0], dtype=np.float64), t_union], axis=0)
    t_union_t = torch.from_numpy(t_union.astype(np.float32)).to(device)
    idx_grid = _map_times_to_indices(t_union, t_grid.astype(np.float64))
    idx_h = _map_times_to_indices(t_union, horizons_arr)

    # Brier accumulators (unweighted)
    brier_sum_grid = np.zeros(t_grid.shape[0], dtype=np.float64)
    brier_sum_h = np.zeros(len(horizons), dtype=np.float64)

    any_pairs = False
    for dt, event_type_1, logits in _iter_eval_pairs(loader, model, device):
        any_pairs = True
        dt_t = dt
        event_type_1_t = event_type_1

        if K is None:
            K = int(logits.shape[1])
            G = int(t_grid.shape[0])
            Hh = int(len(horizons))
            nll_by_k = [[] for _ in range(K)]
            pit_u_by_k = [[] for _ in range(K)]
            brier_sum_grid_by_k = np.zeros((K, G), dtype=np.float64)
            brier_sum_h_by_k = np.zeros((K, Hh), dtype=np.float64)

            cal_bin_count = np.zeros((K, Hh, cal_bins), dtype=np.int64)
            cal_bin_sum_p = np.zeros((K, Hh, cal_bins), dtype=np.float64)
            cal_bin_sum_y = np.zeros((K, Hh, cal_bins), dtype=np.float64)
            citl_pred_sum = np.zeros((K, Hh), dtype=np.float64)
            citl_obs_sum = np.zeros((K, Hh), dtype=np.float64)
            logloss_sum = np.zeros((K, Hh), dtype=np.float64)
            logloss_count = np.zeros((K, Hh), dtype=np.float64)

        # Convert to numpy for IPCW + labels
        t_obs_b = dt_t.detach().cpu().numpy().astype(np.float64)
        # Uncensored next-event data: every instance has an event.
        cens_b = np.zeros_like(t_obs_b, dtype=bool)
        event_ind_b = np.ones_like(t_obs_b, dtype=np.int64)
        n_b = t_obs_b.size

        # NLL per instance
        target_0 = (event_type_1_t - 1).to(torch.long)
        is_cens_t = torch.zeros_like(dt_t, device=device, dtype=torch.bool)
        nll_t = _nll_per_instance(
            loss_type=loss_type,
            logits=logits,
            target_events_0based=target_0,
            t_event=dt_t,
            is_censored=is_cens_t,
            criterion=criterion,
        )
        nll_list.append(nll_t.detach().cpu().numpy().astype(np.float64))

        # Disease-specific NLL buckets
        if nll_by_k is not None and K is not None:
            k_idx = (event_type_1_t.detach().cpu().numpy().astype(np.int64) - 1)
            nll_np = nll_t.detach().cpu().numpy().astype(np.float64)
            for k in np.unique(k_idx):
                if 0 <= int(k) < K:
                    mask = (k_idx == k)
                    if np.any(mask):
                        nll_by_k[int(k)].append(nll_np[mask])

        # Predictions on grid/horizons: p_any(t) = 1 - exp(-Lambda_total(t))
        with torch.no_grad():
            t_grid_t = torch.from_numpy(t_grid.astype(np.float32)).to(device)
            t_h_t = torch.from_numpy(np.asarray(
                horizons, dtype=np.float32)).to(device)

            Lambda_grid = _total_cumhaz_any(
                loss_type, logits, t_grid_t, criterion).detach().cpu().numpy().astype(np.float64)
            Lambda_h = _total_cumhaz_any(
                loss_type, logits, t_h_t, criterion).detach().cpu().numpy().astype(np.float64)

        p_grid = 1.0 - np.exp(-Lambda_grid)
        p_h = 1.0 - np.exp(-Lambda_h)

        p_grid = np.clip(p_grid, 0.0, 1.0)
        p_h = np.clip(p_h, 0.0, 1.0)

        # Labels Y_any(t)
        y_grid = (t_obs_b[:, None] <= t_grid[None, :]).astype(np.float64)
        y_h = (t_obs_b[:, None] <= np.asarray(
            horizons, dtype=np.float64)[None, :]).astype(np.float64)

        # Brier contributions (unweighted)
        brier_sum_grid += np.sum((y_grid - p_grid) ** 2, axis=0)
        brier_sum_h += np.sum((y_h - p_h) ** 2, axis=0)

        # Store horizon arrays for calibration tables and slope
        for j in range(len(horizons)):
            p_h_list[j].append(p_h[:, j].copy())
            y_h_list[j].append(y_h[:, j].copy())
            w_h_list[j].append(np.ones_like(y_h[:, j], dtype=np.float64))

        # Disease-specific CIFs on union time grid (grid + horizons)
        if K is not None and brier_sum_grid_by_k is not None and brier_sum_h_by_k is not None:
            # Pre-calculate y_time_grid and y_time_h for the batch (M, G) and (M, H)
            y_time_grid = (t_obs_b[:, None] <=
                           t_grid[None, :]).astype(np.float64)
            y_time_h = (t_obs_b[:, None] <=
                        horizons_arr[None, :]).astype(np.float64)

            k_idx = (event_type_1_t.detach().cpu(
            ).numpy().astype(np.int64) - 1)  # (M,)

            # Iterate over chunks of diseases
            for k0, k1, cif_chunk_t in _predict_cif_chunked(
                loss_type=loss_type,
                logits=logits,
                t_eval=t_union_t,
                criterion=criterion,
                k_chunk=k_chunk,
                use_amp=use_amp_eval,
            ):
                # cif_chunk_t: (M, Kc, TU)
                Kc = k1 - k0

                # Slice for grid and horizons
                cif_grid_chunk = cif_chunk_t[:, :, idx_grid].float(
                ).cpu().numpy()  # (M, Kc, G)
                cif_h_chunk = cif_chunk_t[:, :, idx_h].float(
                ).cpu().numpy()       # (M, Kc, H)

                # Move full chunk to CPU for PIT if needed
                cif_chunk_cpu = cif_chunk_t.float().cpu().numpy()  # (M, Kc, TU)

                del cif_chunk_t
                if empty_cache_debug:
                    torch.cuda.empty_cache()

                # --- CPU Processing ---

                # Event indicators for this chunk: 1{event == k}
                disease_range = np.arange(k0, k1)
                mask_chunk = (k_idx[:, None] == disease_range[None, :]).astype(
                    np.float64)  # (M, Kc)

                # Brier Score Updates
                y_grid_k_chunk = mask_chunk[:, :, None] * \
                    y_time_grid[:, None, :]  # (M, Kc, G)
                diff_grid = y_grid_k_chunk - cif_grid_chunk
                brier_sum_grid_by_k[k0:k1, :] += np.sum(diff_grid**2, axis=0)

                y_h_k_chunk = mask_chunk[:, :, None] * \
                    y_time_h[:, None, :]  # (M, Kc, H)
                diff_h = y_h_k_chunk - cif_h_chunk
                brier_sum_h_by_k[k0:k1, :] += np.sum(diff_h**2, axis=0)

                # CITL, Logloss, Calibration Bins
                for j in range(len(horizons)):
                    # Handle potential NaNs from AMP or numerical instability
                    raw_p = cif_h_chunk[:, :, j]
                    if np.any(np.isnan(raw_p)):
                        raw_p = np.nan_to_num(raw_p, nan=0.0)
                    p_j = np.clip(raw_p, 0.0, 1.0)  # (M, Kc)
                    y_j = y_h_k_chunk[:, :, j]                    # (M, Kc)

                    citl_pred_sum[k0:k1, j] += p_j.sum(axis=0)
                    citl_obs_sum[k0:k1, j] += y_j.sum(axis=0)

                    p_clip = np.clip(p_j, 1e-6, 1.0 - 1e-6)
                    ll = -(y_j * np.log(p_clip) + (1.0 - y_j)
                           * np.log(1.0 - p_clip))
                    logloss_sum[k0:k1, j] += ll.sum(axis=0)
                    logloss_count[k0:k1, j] += float(n_b)

                    # Binning
                    bin_idx = np.minimum(
                        (p_j * cal_bins).astype(np.int64), cal_bins - 1)  # (M, Kc)

                    for kk_local in range(Kc):
                        k_global = k0 + kk_local
                        idxk = bin_idx[:, kk_local]

                        cal_bin_count[k_global, j,
                                      :] += np.bincount(idxk, minlength=cal_bins)
                        cal_bin_sum_p[k_global, j, :] += np.bincount(
                            idxk, weights=p_j[:, kk_local], minlength=cal_bins)
                        cal_bin_sum_y[k_global, j, :] += np.bincount(
                            idxk, weights=y_j[:, kk_local], minlength=cal_bins)

                # Conditional PIT
                # For each disease k in this chunk, if we have events for k in this batch, compute PIT.
                tmax_idx = -1

                # Identify diseases in this chunk that have events in this batch
                batch_diseases = np.unique(k_idx)
                chunk_diseases = batch_diseases[(
                    batch_diseases >= k0) & (batch_diseases < k1)]

                for k in chunk_diseases:
                    kk = int(k)
                    kk_local = kk - k0
                    sel = (k_idx == kk)

                    cif_series = cif_chunk_cpu[sel, kk_local, :]  # (Nk, TU)
                    cif_at_t = _interp1d_batch(
                        t_union, cif_series, t_obs_b[sel])
                    pi = np.clip(cif_series[:, tmax_idx], 1e-12, 1.0)
                    u_cond = np.clip(cif_at_t / pi, 0.0, 1.0)
                    pit_u_by_k[kk].append(u_cond.astype(np.float64))

        # PIT (all are events): u = F(T_obs)
        with torch.no_grad():
            Lambda_obs = _total_cumhaz_any_per_instance(
                loss_type=loss_type,
                logits=logits,
                t_event=torch.from_numpy(t_obs_b.astype(np.float32)).to(device),
                criterion=criterion,
            ).detach().cpu().numpy().astype(np.float64)
        u = np.clip(1.0 - np.exp(-Lambda_obs), 0.0, 1.0)
        pit_u.append(u)

        n_total += n_b

    if not any_pairs or n_total == 0:
        raise RuntimeError("No valid evaluation pairs found.")

    print("[2/5] Aggregating global metrics...")

    # Aggregate
    nll_all = np.concatenate(nll_list, axis=0)
    nll_mean = float(np.mean(nll_all))
    nll_median = float(np.median(nll_all))
    nll_p95 = float(np.quantile(nll_all, 0.95))

    brier_grid = brier_sum_grid / float(n_total)
    brier_h = brier_sum_h / float(n_total)

    # IBS via trapezoid
    ibs_any = float(np.trapezoid(brier_grid, t_grid) / float(t_grid[-1]))

    # CITL + cal slope per horizon
    metrics: Dict[str, float] = {
        "nll_mean": nll_mean,
        "nll_median": nll_median,
        "nll_p95": nll_p95,
        "ibs_any": ibs_any,
    }

    # PIT stats
    if len(pit_u) > 0:
        u_all = np.concatenate(pit_u, axis=0)
    else:
        u_all = np.array([], dtype=np.float64)

    pit_ks_stat, pit_ks_pvalue = _ks_uniform(u_all)
    metrics["pit_ks_stat"] = float(pit_ks_stat)
    metrics["pit_ks_pvalue"] = float(pit_ks_pvalue)
    metrics["pit_mean"] = float(
        np.mean(u_all)) if u_all.size > 0 else float("nan")
    metrics["pit_var"] = float(
        np.var(u_all)) if u_all.size > 0 else float("nan")

    # Save PIT histogram
    hist_counts, bin_edges = np.histogram(
        u_all, bins=pit_bins, range=(0.0, 1.0))
    pit_df = pd.DataFrame({
        "bin_left": bin_edges[:-1],
        "bin_right": bin_edges[1:],
        "count": hist_counts,
    })
    pit_df.to_csv(os.path.join(out_dir, "pit_hist.csv"), index=False)

    print("[3/5] Writing any-event calibration tables + slopes...")

    # Save calibration tables + compute slope/CITL
    def _hkey(H: float) -> str:
        # Defaults: 1,5,10 -> h1,h5,h10 (matches requested JSON keys)
        if abs(H - round(H)) < 1e-9:
            return f"h{int(round(H))}"
        s = ("%g" % H).replace(".", "p")
        return f"h{s}"

    for j, H in enumerate(horizons):
        hk = _hkey(float(H))
        pH = np.concatenate(p_h_list[j], axis=0).astype(np.float64)
        yH = np.concatenate(y_h_list[j], axis=0).astype(np.float64)
        wH = np.concatenate(w_h_list[j], axis=0).astype(np.float64)

        # Brier scalar
        metrics[f"brier_any_{hk}"] = float(brier_h[j])

        # Binary log loss (within-horizon event NLL)
        # y_H = 1{dt <= H}, p_H = F_any(H)
        p_clip = np.clip(pH, 1e-12, 1.0 - 1e-12)
        logloss = -float(np.mean(yH * np.log(p_clip) +
                         (1.0 - yH) * np.log(1.0 - p_clip)))
        metrics[f"logloss_any_{hk}"] = logloss

        # CITL: pred mean - observed event rate (unweighted)
        pred = float(np.mean(pH))
        obs = float(np.mean(yH)) if yH.size > 0 else float("nan")
        metrics[f"citl_any_{hk}"] = float(pred - obs)

        # Calibration slope
        try:
            slope = compute_cal_slope_weighted_logistic(
                y=yH, p=pH, w=np.ones_like(yH, dtype=np.float64)
            )
        except Exception:
            slope = float("nan")
        metrics[f"cal_slope_any_{hk}"] = float(slope)

        # Decile table
        df = pd.DataFrame({"p": pH, "y": yH})
        # qcut can drop bins if too many ties; fall back to rank-based bins
        try:
            df["decile"] = pd.qcut(df["p"], 10, labels=False, duplicates="drop")
        except Exception:
            df["decile"] = pd.qcut(df["p"].rank(
                method="average"), 10, labels=False, duplicates="drop")

        calib = (
            df.groupby("decile", dropna=False)
            .agg(
                count=("p", "size"),
                mean_pred=("p", "mean"),
                obs_rate=("y", "mean"),
            )
            .reset_index()
            .sort_values("decile")
        )
        calib.to_csv(os.path.join(
            out_dir, f"calibration_table_{hk}.csv"), index=False)

    # -------------------------
    # Disease-specific summary CSVs
    # -------------------------
    if K is not None and nll_by_k is not None and brier_sum_grid_by_k is not None and brier_sum_h_by_k is not None:
        print("[4/5] Computing disease-specific summaries...")
        brier_grid_by_k = brier_sum_grid_by_k / float(n_total)  # (K,G)
        brier_h_by_k = brier_sum_h_by_k / float(n_total)  # (K,H)

        # IBS per disease
        ibs_by_k = np.trapezoid(
            brier_grid_by_k, t_grid[None, :], axis=1) / float(t_grid[-1])

        rows: List[Dict[str, float]] = []
        pit_hist_rows: List[Dict[str, float]] = []
        calib_rows: List[Dict[str, float]] = []

        for k in _progress(range(K), desc="Per-disease metrics", total=K, leave=False):
            # NLL stats for events of this disease
            if len(nll_by_k[k]) > 0:
                nll_k = np.concatenate(nll_by_k[k], axis=0)
                nll_mean_k = float(np.mean(nll_k))
                nll_median_k = float(np.median(nll_k))
                nll_p95_k = float(np.quantile(nll_k, 0.95))
                n_events = int(nll_k.size)
            else:
                nll_mean_k = float("nan")
                nll_median_k = float("nan")
                nll_p95_k = float("nan")
                n_events = 0

            row: Dict[str, float] = {
                "disease_index_1based": float(k + 1),
                "n_events": float(n_events),
                "nll_mean": nll_mean_k,
                "nll_median": nll_median_k,
                "nll_p95": nll_p95_k,
                "ibs": float(ibs_by_k[k]),
            }

            # PIT conditional stats + histogram
            if pit_u_by_k is not None and len(pit_u_by_k[k]) > 0:
                u_k = np.concatenate(pit_u_by_k[k], axis=0)
            else:
                u_k = np.array([], dtype=np.float64)

            pit_ks_stat_k, pit_ks_pvalue_k = _ks_uniform(u_k)
            row["pit_ks_stat"] = float(pit_ks_stat_k)
            row["pit_ks_pvalue"] = float(pit_ks_pvalue_k)
            row["pit_mean"] = float(
                np.mean(u_k)) if u_k.size > 0 else float("nan")
            row["pit_var"] = float(
                np.var(u_k)) if u_k.size > 0 else float("nan")

            hist_counts_k, bin_edges_k = np.histogram(
                u_k, bins=pit_bins, range=(0.0, 1.0))
            for bi in range(pit_bins):
                pit_hist_rows.append({
                    "disease_index_1based": float(k + 1),
                    "bin_left": float(bin_edges_k[bi]),
                    "bin_right": float(bin_edges_k[bi + 1]),
                    "count": float(hist_counts_k[bi]),
                })

            # Horizon metrics + calibration summaries
            for j, H in enumerate(horizons):
                hk = (f"h{int(round(float(H)))}" if abs(
                    float(H) - round(float(H))) < 1e-9 else f"h{('%g' % float(H)).replace('.', 'p')}")
                row[f"brier_{hk}"] = float(brier_h_by_k[k, j])

                if citl_pred_sum is not None and citl_obs_sum is not None and logloss_sum is not None and logloss_count is not None:
                    pred_mean = float(citl_pred_sum[k, j] / float(n_total))
                    obs_mean = float(citl_obs_sum[k, j] / float(n_total))
                    row[f"citl_{hk}"] = float(pred_mean - obs_mean)
                    row[f"logloss_{hk}"] = float(
                        logloss_sum[k, j] / max(float(logloss_count[k, j]), 1.0))

                # Calibration slope from binned summaries
                slope = float("nan")
                if cal_bin_count is not None and cal_bin_sum_p is not None and cal_bin_sum_y is not None:
                    cnt = cal_bin_count[k, j, :].astype(np.float64)
                    nonzero = cnt > 0
                    if np.any(nonzero):
                        nz_idx = np.where(nonzero)[0]
                        mean_p = (
                            cal_bin_sum_p[k, j, nonzero] / cnt[nonzero]).astype(np.float64)
                        obs_r = (cal_bin_sum_y[k, j, nonzero] /
                                 cnt[nonzero]).astype(np.float64)
                        w = cnt[nonzero].astype(np.float64)
                        try:
                            slope = compute_cal_slope_weighted_logistic(
                                y=obs_r, p=mean_p, w=w)
                        except Exception:
                            slope = float("nan")
                        for b_idx, c, mp, orr in zip(nz_idx, cnt[nonzero], mean_p, obs_r):
                            calib_rows.append({
                                "disease_index_1based": float(k + 1),
                                "horizon": float(H),
                                "bin": float(b_idx),
                                "count": float(c),
                                "mean_pred": float(mp),
                                "obs_rate": float(orr),
                            })

                row[f"cal_slope_{hk}"] = float(slope)

            rows.append(row)

        metrics_df = pd.DataFrame(rows)
        metrics_df.to_csv(os.path.join(
            out_dir, "metrics_by_disease.csv"), index=False)

        pit_hist_by_disease_df = pd.DataFrame(pit_hist_rows)
        pit_hist_by_disease_df.to_csv(os.path.join(
            out_dir, "pit_hist_by_disease.csv"), index=False)

        calib_df = pd.DataFrame(calib_rows)
        calib_df.to_csv(os.path.join(
            out_dir, "calibration_by_disease.csv"), index=False)

    # Save summary
    print("[5/5] Saving summary metrics...")
    with open(os.path.join(out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate likelihood-consistent survival metrics")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to a run directory under runs/*")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path (default: best_model.pt in run_dir)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--t_max", type=float, default=10.0,
                        help="Max horizon for IBS grid (years)")
    parser.add_argument("--n_grid", type=int, default=100,
                        help="Number of points in IBS time grid")
    parser.add_argument("--horizons", type=str, default="1,5,10",
                        help="Comma-separated horizon years")

    parser.add_argument("--pit_bins", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    # Performance / Memory args
    parser.add_argument("--k_chunk", type=int, default=64,
                        help="Chunk size for disease dimension to avoid OOM")
    parser.add_argument("--t_chunk", type=int, default=0,
                        help="Chunk size for time dimension (0=disabled)")
    parser.add_argument("--use_amp_eval", type=str, default="False",
                        help="Enable AMP for evaluation (True/False)")
    parser.add_argument("--empty_cache_debug", type=str, default="False",
                        help="Force empty_cache() between chunks (slow, for debug)")

    args = parser.parse_args()

    # Parse bools
    use_amp_eval = args.use_amp_eval.lower() == "true"
    empty_cache_debug = args.empty_cache_debug.lower() == "true"

    run_dir = args.run_dir
    print("[Step] Loading training config...")
    train_cfg = _load_train_config(run_dir)

    if args.checkpoint is not None:
        checkpoint = args.checkpoint
    else:
        # Match train.py output filenames; keep a fallback for older runs.
        candidate = os.path.join(run_dir, "best_model.pt")
        fallback = os.path.join(run_dir, "best_checkpoint.pt")
        checkpoint = candidate if os.path.exists(candidate) else fallback
    out_dir = run_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Step] Run dir: {run_dir}")
    print(f"[Step] Checkpoint: {checkpoint}")
    print(f"[Step] Output dir: {out_dir}")

    device = torch.device(args.device)
    print(f"[Step] Device: {device}")

    # Build dataset consistent with training
    print("[Step] Building dataset + test split...")
    full_cov = bool(train_cfg.get("full_cov", False))
    cov_list = None if full_cov else ["bmi", "smoking", "alcohol"]

    data_prefix = str(train_cfg.get("data_prefix", "ukb"))
    dataset = HealthDataset(data_prefix=data_prefix, covariate_list=cov_list)

    # Split with same ratios/seed
    n_total = len(dataset)
    train_ratio = float(train_cfg.get("train_ratio", 0.7))
    val_ratio = float(train_cfg.get("val_ratio", 0.15))
    test_ratio = float(train_cfg.get("test_ratio", 0.15))
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    gen = torch.Generator().manual_seed(int(train_cfg.get("random_seed", args.seed)))
    train_data, val_data, test_data = random_split(
        dataset, [n_train, n_val, n_test], generator=gen)

    print(
        f"[Step] Split sizes: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    loader = DataLoader(
        test_data,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=health_collate_fn,
    )

    try:
        nb = len(loader)
    except Exception:
        nb = None
    print(
        f"[Step] DataLoader ready (batch_size={int(args.batch_size)}, num_batches={nb})")

    print("[Step] Building model + loss...")
    model, criterion = _build_model_and_loss(train_cfg, dataset, device)

    print("[Step] Loading checkpoint weights...")
    _load_checkpoint(model, checkpoint, device)

    # Grid and horizons
    print("[Step] Preparing time grid + horizons...")
    horizons = tuple(float(x)
                     for x in str(args.horizons).split(",") if x.strip() != "")
    t_grid = np.linspace(0.0, float(args.t_max),
                         int(args.n_grid), dtype=np.float64)
    # Avoid a large flat at exactly 0 for log-time hazards
    if t_grid.size > 1:
        t_grid[1] = max(t_grid[1], 1e-6)

    metrics = compute_metrics(
        model=model,
        criterion=criterion,
        loader=loader,
        loss_type=str(train_cfg["loss_type"]),
        t_grid=t_grid,
        horizons=horizons,
        out_dir=out_dir,
        pit_bins=int(args.pit_bins),
        seed=int(args.seed),
        k_chunk=int(args.k_chunk),
        t_chunk=int(args.t_chunk),
        use_amp_eval=use_amp_eval,
        empty_cache_debug=empty_cache_debug,
    )

    print("[Done] Metrics computed. Summary:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
