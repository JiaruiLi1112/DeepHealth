"""Evaluate clinical/business utility metrics for DelphiFork.

This script is intentionally structured to mirror `evaluate_likelihood_metrics.py`:
- argparse interface (run_dir/checkpoint/device/batch_size/horizons/seed)
- model + loss construction via `_build_model_and_loss`
- checkpoint loading
- dataset construction + test split logic
- `_iter_eval_pairs` inference loop

Clinical utility outputs (focused on health-checkup + insurance use cases):
1) Discrimination
   - Harrell's C-index (ranking power)
   - Time-dependent AUC at horizons (default: 1,3,5,10 years)
2) Risk stratification / enrichment at 5y
   - High (top 10%), Medium, Low (bottom 50%)
   - Enrichment factor = event_rate(top10%) / event_rate(overall)
   - CSV: risk_stratification.csv
3) Decision Curve Analysis (DCA) at 5y
   - Net benefit for thresholds in [0, 0.5]
   - CSV: dca_values.csv
4) Counterfactual intervention simulation
   - Pick top 100 highest-risk instances (5y)
   - Reduce BMI by 10%, set smoking to non-smoker
   - Report ARR / RRR

Important note about evaluation instances:
This repo's evaluation paradigm treats each valid (prev -> next) event pair as an
instance with observed next-event time `dt` (years) (see `get_valid_pairs_and_dt`).
That means there is no censoring in the extracted pair dataset. Metrics that
traditionally account for censoring (C-index, time-dependent AUC, DCA) are
therefore computed on fully observed pair outcomes here.

Outputs:
- clinical_metrics_summary.json
- risk_stratification.csv
- dca_values.csv

Usage:
  python evaluate_clinical_metrics.py --run_dir runs/<your_run>
"""

from __future__ import annotations

import argparse
import json
import os
import heapq
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
)
from model import DelphiFork


# -------------------------
# Progress helper (template-compatible)
# -------------------------


def _progress(iterable, *, desc: str, total: Optional[int] = None, leave: bool = False):
    """Lightweight wrapper: uses tqdm if available, else returns iterable."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=leave)


# -------------------------
# Config / model loading (template-compatible)
# -------------------------


def _load_train_config(run_dir: str) -> Dict:
    cfg_path = os.path.join(run_dir, "train_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_model_and_loss(cfg: Dict, dataset: HealthDataset, device: torch.device):
    """Match training-time model/loss construction."""
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
        # local import to avoid circular import at module import time
        from train import bin_edges

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


# -------------------------
# Parametric survival helper
# -------------------------


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
        t: (H,) times in years

    Returns:
        total_cumhaz: (M,H)
    """
    t = t.to(device=logits.device)
    zero_mask = t <= 0
    t_safe = torch.where(zero_mask, torch.as_tensor(
        eps, device=t.device, dtype=t.dtype), t)

    if loss_type == "exponential":
        if logits.dim() == 3:
            logits = logits.squeeze(-1)
        hazards = F.softplus(logits) + eps  # (M,K)
        total_hazard = hazards.sum(dim=1)  # (M,)
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
        log_t = torch.log(t_safe).view(1, 1, H)
        total = torch.zeros((M, H), device=logits.device, dtype=logits.dtype)
        for k0 in range(0, K, k_chunk):
            k1 = min(K, k0 + k_chunk)
            sh = shapes[:, k0:k1].unsqueeze(-1)
            sc = scales[:, k0:k1].unsqueeze(-1)
            contrib = sc * torch.exp(sh * log_t)
            total = total + contrib.sum(dim=1)
        if torch.any(zero_mask):
            total = total.clone()
            total[:, zero_mask] = 0.0
        return total

    if loss_type == "lognormal":
        # Criterion method expects (M,H)
        t_mat = t_safe.view(1, -1).expand(logits.shape[0], -1)
        total = criterion.predict_total_cum_hazard(logits, t_mat)
        if torch.any(zero_mask):
            total = total.clone()
            total[:, zero_mask] = 0.0
        return total

    raise ValueError(f"Unsupported loss_type: {loss_type}")


def _risk_from_logits(
    *,
    loss_type: str,
    logits: torch.Tensor,
    horizons_years: Sequence[float],
    criterion: torch.nn.Module,
) -> np.ndarray:
    """Return risk(t)=1-S(t) for each horizon. Output shape: (M,H)."""
    device = logits.device
    t_h = torch.as_tensor(np.asarray(
        horizons_years, dtype=np.float32), device=device)
    Lambda = _total_cumhaz_any(loss_type, logits, t_h, criterion)
    risk = 1.0 - torch.exp(-Lambda)
    risk = torch.clamp(risk, 0.0, 1.0)
    return risk.detach().cpu().numpy().astype(np.float64)


# -------------------------
# Evaluation loop (template-compatible)
# -------------------------


def _iter_eval_pairs(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]]:
    """Yield (dt_years, event_type_1based, logits, payload) for each batch of extracted pairs.

    payload contains (event_seq, time_seq, cont_feats, cate_feats, sexes, b_prev, t_prev)
    and is used only for counterfactual simulation.
    """
    model.eval()
    with torch.no_grad():
        try:
            total_batches: Optional[int] = len(loader)
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
            target_events_0 = (event_seq[b_next, t_next] - 2).to(torch.long)
            event_type_1 = target_events_0 + 1
            payload = (event_seq, time_seq, cont_feats,
                       cate_feats, sexes, b_prev, t_prev)
            yield dt, event_type_1, logits, payload


# -------------------------
# Metric implementations
# -------------------------


class _Fenwick:
    def __init__(self, n: int):
        self.n = int(n)
        self.bit = np.zeros(self.n + 1, dtype=np.int64)

    def add(self, idx1: int, delta: int = 1) -> None:
        i = int(idx1)
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i

    def sum(self, idx1: int) -> int:
        s = 0
        i = int(idx1)
        while i > 0:
            s += int(self.bit[i])
            i -= i & -i
        return int(s)


def _harrell_c_index(times: np.ndarray, scores: np.ndarray) -> float:
    """Harrell's C-index for fully observed event times.

    Higher score should indicate higher risk (shorter time).
    This implementation ignores censoring (consistent with pair-eval data).
    """
    times = np.asarray(times, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(times) & np.isfinite(scores)
    times = times[mask]
    scores = scores[mask]
    n = times.size
    if n < 2:
        return float("nan")

    order = np.argsort(times, kind="mergesort")
    times = times[order]
    scores = scores[order]

    # Score ranks (1..m) for Fenwick; higher score means higher risk.
    uniq_scores = np.unique(scores)
    score_rank = np.searchsorted(uniq_scores, scores) + 1

    bit = _Fenwick(int(uniq_scores.size))
    concordant = 0.0
    comparable = 0.0

    i = 0
    while i < n:
        j = i
        # group by identical event time (ties in time are not comparable)
        while j < n and times[j] == times[i]:
            j += 1

        prev_count = i
        if prev_count > 0:
            for k in range(i, j):
                r = int(score_rank[k])
                leq = bit.sum(r)
                lt = bit.sum(r - 1)
                eq = leq - lt
                gt = prev_count - leq
                comparable += prev_count
                concordant += gt + 0.5 * eq

        # now add this group's scores to BIT
        for k in range(i, j):
            bit.add(int(score_rank[k]), 1)

        i = j

    if comparable <= 0:
        return float("nan")
    return float(concordant / comparable)


def _roc_auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC with sklearn if available, else a tie-aware rank formula."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    mask = np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    try:
        from sklearn.metrics import roc_auc_score  # type: ignore

        return float(roc_auc_score(y_true, y_score))
    except Exception:
        # Mann–Whitney U with tie-aware average ranks
        order = np.argsort(y_score, kind="mergesort")
        y_score_sorted = y_score[order]
        y_true_sorted = y_true[order]

        ranks = np.empty_like(y_score_sorted, dtype=np.float64)
        i = 0
        r = 1
        while i < y_score_sorted.size:
            j = i
            while j < y_score_sorted.size and y_score_sorted[j] == y_score_sorted[i]:
                j += 1
            # average rank for tie block [i, j)
            avg_rank = 0.5 * ((r) + (r + (j - i) - 1))
            ranks[i:j] = avg_rank
            r += (j - i)
            i = j

        sum_pos_ranks = float(np.sum(ranks[y_true_sorted == 1]))
        auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(auc)


def _infer_feature_indices(
    *,
    data_prefix: str,
    covariate_list: Optional[List[str]],
) -> Tuple[Optional[int], Optional[int]]:
    """Infer BMI continuous index and smoking categorical index using dataset's logic.

    This re-runs the same float/int + unique-count heuristic used by `HealthDataset`.
    Returns (bmi_cont_idx, smoking_cate_idx), each optional.
    """
    table_path = f"{data_prefix}_table.csv"
    if not os.path.exists(table_path):
        return None, None

    tabular_data = pd.read_csv(table_path, index_col="eid").convert_dtypes()
    if covariate_list is not None:
        # keep exact order
        tabular_data = tabular_data[covariate_list]

    cont_cols: List[str] = []
    cate_cols: List[str] = []
    for col in tabular_data.columns:
        if pd.api.types.is_float_dtype(tabular_data[col]):
            cont_cols.append(col)
        elif pd.api.types.is_integer_dtype(tabular_data[col]):
            series = tabular_data[col]
            unique_vals = series.dropna().unique()
            if len(unique_vals) > 11:
                cont_cols.append(col)
            else:
                cate_cols.append(col)

    bmi_cont_idx = cont_cols.index("bmi") if "bmi" in cont_cols else None
    smoking_cate_idx = cate_cols.index(
        "smoking") if "smoking" in cate_cols else None
    return bmi_cont_idx, smoking_cate_idx


@dataclass
class _TopInstance:
    risk_5y: float
    event_seq: torch.Tensor
    time_seq: torch.Tensor
    sex: torch.Tensor
    cont_seq: torch.Tensor
    cate_seq: torch.Tensor
    t_prev: int


def compute_clinical_metrics(
    *,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    loader: DataLoader,
    loss_type: str,
    horizons: Sequence[float],
    out_dir: str,
    seed: int,
    business_horizon: float = 5.0,
    dca_threshold_max: float = 0.5,
    dca_threshold_steps: int = 51,
    topk_intervention: int = 100,
    bmi_cont_idx: Optional[int] = None,
    smoking_cate_idx: Optional[int] = None,
    smoking_non_smoker_value: int = 1,
) -> Dict[str, object]:
    device = next(model.parameters()).device
    rng = np.random.default_rng(int(seed))
    # deterministic placeholder; keeps signature consistent if extended.
    _ = rng

    horizons = tuple(float(h) for h in horizons)
    if len(horizons) == 0:
        raise ValueError("horizons must be non-empty")

    # Ensure business horizon is included for downstream computations.
    all_horizons = list(horizons)
    if float(business_horizon) not in all_horizons:
        all_horizons.append(float(business_horizon))
    all_horizons = tuple(sorted(set(all_horizons)))

    dt_all: List[np.ndarray] = []
    risk_by_h: Dict[float, List[np.ndarray]] = {h: [] for h in all_horizons}

    # Maintain top-K by risk at business horizon for counterfactual inference.
    # Heap stores (risk, unique_id, _TopInstance)
    top_heap: List[Tuple[float, int, _TopInstance]] = []
    uid = 0

    any_pairs = False
    for dt, _event_type_1, logits, payload in _iter_eval_pairs(loader, model, device):
        any_pairs = True
        dt_np = dt.detach().cpu().numpy().astype(np.float64)
        dt_all.append(dt_np)

        risk_mat = _risk_from_logits(
            loss_type=loss_type,
            logits=logits,
            horizons_years=all_horizons,
            criterion=criterion,
        )

        for j, h in enumerate(all_horizons):
            risk_by_h[h].append(risk_mat[:, j])

        # Counterfactual: consider top-K instances by risk at business horizon.
        h_idx = all_horizons.index(float(business_horizon))
        risk_5 = risk_mat[:, h_idx]
        event_seq, time_seq, cont_feats, cate_feats, sexes, b_prev, t_prev = payload

        # Store per-instance single-sample payload on CPU (only for top-K)
        for i in range(risk_5.shape[0]):
            r = float(risk_5[i])
            if not np.isfinite(r):
                continue
            b = int(b_prev[i].detach().cpu().item())
            tp = int(t_prev[i].detach().cpu().item())

            inst = _TopInstance(
                risk_5y=r,
                event_seq=event_seq[b: b + 1].detach().cpu(),
                time_seq=time_seq[b: b + 1].detach().cpu(),
                sex=sexes[b: b + 1].detach().cpu(),
                cont_seq=cont_feats[b: b + 1].detach().cpu(),
                cate_seq=cate_feats[b: b + 1].detach().cpu(),
                t_prev=tp,
            )
            if len(top_heap) < int(topk_intervention):
                heapq.heappush(top_heap, (r, uid, inst))
                uid += 1
            else:
                # Replace smallest if current is larger
                if r > top_heap[0][0]:
                    heapq.heapreplace(top_heap, (r, uid, inst))
                    uid += 1

    if not any_pairs:
        raise RuntimeError(
            "No valid (prev->next) pairs were extracted from the test split.")

    dt_all_np = np.concatenate(dt_all, axis=0)
    risk_np_by_h: Dict[float, np.ndarray] = {
        h: np.concatenate(v, axis=0) for h, v in risk_by_h.items()}

    # -------------------------
    # Discrimination: C-index + time-dependent AUCs
    # -------------------------
    score_for_c = risk_np_by_h[float(business_horizon)]
    c_index = _harrell_c_index(dt_all_np, score_for_c)

    aucs: Dict[str, float] = {}
    for h in horizons:
        risk_h = risk_np_by_h[float(h)]
        y_h = (dt_all_np <= float(h)).astype(np.int64)
        aucs[f"auc_{int(h) if float(h).is_integer() else h}y"] = _roc_auc_safe(
            y_h, risk_h)

    # -------------------------
    # Risk stratification / enrichment (business horizon)
    # -------------------------
    risk_5 = risk_np_by_h[float(business_horizon)]
    y_5 = (dt_all_np <= float(business_horizon)).astype(np.int64)

    valid = np.isfinite(risk_5) & np.isfinite(dt_all_np)
    risk_5_v = risk_5[valid]
    y_5_v = y_5[valid]

    n = risk_5_v.size
    if n == 0:
        enrichment_factor = float("nan")
        strat_df = pd.DataFrame(
            {"risk_group": ["High", "Medium", "Low"], "observed_event_rate": [
                np.nan, np.nan, np.nan], "predicted_risk": [np.nan, np.nan, np.nan]}
        )
    else:
        order = np.argsort(-risk_5_v, kind="mergesort")
        n_high = max(1, int(np.ceil(0.10 * n)))
        n_low = max(1, int(np.floor(0.50 * n)))
        if n_high + n_low > n:
            n_low = max(0, n - n_high)

        idx_high = order[:n_high]
        idx_low = order[n -
                        n_low:] if n_low > 0 else np.array([], dtype=np.int64)
        idx_medium = order[n_high: n -
                           n_low] if (n - n_low) > n_high else np.array([], dtype=np.int64)

        rate_all = float(np.mean(y_5_v)) if y_5_v.size > 0 else float("nan")
        rate_high = float(np.mean(y_5_v[idx_high])
                          ) if idx_high.size > 0 else float("nan")
        enrichment_factor = float(rate_high / rate_all) if (np.isfinite(
            rate_high) and np.isfinite(rate_all) and rate_all > 0) else float("nan")

        rows = []
        for name, idx in [("High", idx_high), ("Medium", idx_medium), ("Low", idx_low)]:
            if idx.size == 0:
                rows.append(
                    {"risk_group": name, "observed_event_rate": np.nan, "predicted_risk": np.nan})
            else:
                rows.append(
                    {
                        "risk_group": name,
                        "observed_event_rate": float(np.mean(y_5_v[idx])),
                        "predicted_risk": float(np.mean(risk_5_v[idx])),
                    }
                )
        strat_df = pd.DataFrame(rows)

    strat_path = os.path.join(out_dir, "risk_stratification.csv")
    strat_df.to_csv(strat_path, index=False)

    # -------------------------
    # Decision Curve Analysis (DCA) at business horizon
    # -------------------------
    thr = np.linspace(0.0, float(dca_threshold_max), int(
        dca_threshold_steps), dtype=np.float64)
    dca_rows: List[Dict[str, float]] = []
    N = float(y_5_v.size)
    event_rate = float(np.mean(y_5_v)) if y_5_v.size > 0 else float("nan")

    for t in thr:
        if y_5_v.size == 0:
            nb_model = float("nan")
            nb_all = float("nan")
            nb_none = 0.0
        else:
            pred_pos = risk_5_v >= t
            tp = float(np.sum((pred_pos) & (y_5_v == 1)))
            fp = float(np.sum((pred_pos) & (y_5_v == 0)))
            # Net benefit formula
            if t >= 1.0:
                w = float("inf")
            else:
                w = float(t / max(1e-12, (1.0 - t)))
            nb_model = (tp / N) - (fp / N) * w
            # Treat-all: everyone positive
            nb_all = event_rate - (1.0 - event_rate) * w
            nb_none = 0.0

        dca_rows.append(
            {
                "threshold": float(t),
                "net_benefit_model": float(nb_model),
                "net_benefit_all": float(nb_all),
                "net_benefit_none": float(nb_none),
            }
        )

    dca_df = pd.DataFrame(dca_rows)
    dca_path = os.path.join(out_dir, "dca_values.csv")
    dca_df.to_csv(dca_path, index=False)

    # -------------------------
    # Counterfactual intervention simulation
    # -------------------------
    # Top instances are stored as a min-heap; convert to descending risk list.
    top_instances = [x[2]
                     for x in sorted(top_heap, key=lambda z: z[0], reverse=True)]

    baseline_risks: List[float] = []
    post_risks: List[float] = []

    model.eval()
    with torch.no_grad():
        for inst in top_instances:
            # Move to device
            event_seq = inst.event_seq.to(device)
            time_seq = inst.time_seq.to(device)
            sex = inst.sex.to(device)
            cont_seq = inst.cont_seq.to(device)
            cate_seq = inst.cate_seq.to(device)
            t_prev = int(inst.t_prev)

            b_prev = torch.zeros((1,), device=device, dtype=torch.long)
            t_prev_t = torch.as_tensor(
                [t_prev], device=device, dtype=torch.long)

            # Baseline prediction
            logits_base = model(event_seq, time_seq, sex, cont_seq,
                                cate_seq, b_prev=b_prev, t_prev=t_prev_t)
            risk_base = float(
                _risk_from_logits(
                    loss_type=loss_type,
                    logits=logits_base,
                    horizons_years=[float(business_horizon)],
                    criterion=criterion,
                )[0, 0]
            )
            baseline_risks.append(risk_base)

            # Intervention: modify features in memory
            cont_mod = cont_seq.clone()
            cate_mod = cate_seq.clone()

            if bmi_cont_idx is not None:
                if cont_mod.dim() == 3 and cont_mod.size(-1) > int(bmi_cont_idx):
                    bmi_val = cont_mod[0, 0, int(bmi_cont_idx)]
                    if torch.isfinite(bmi_val):
                        cont_mod[0, 0, int(bmi_cont_idx)] = 0.9 * bmi_val

            if smoking_cate_idx is not None:
                if cate_mod.dim() == 3 and cate_mod.size(-1) > int(smoking_cate_idx):
                    cur = int(cate_mod[0, 0, int(smoking_cate_idx)].item())
                    # Only override if it's a known (non-missing) value and not already non-smoker.
                    if cur > 0 and cur != int(smoking_non_smoker_value):
                        cate_mod[0, 0, int(smoking_cate_idx)] = int(
                            smoking_non_smoker_value)

            logits_post = model(event_seq, time_seq, sex, cont_mod,
                                cate_mod, b_prev=b_prev, t_prev=t_prev_t)
            risk_post = float(
                _risk_from_logits(
                    loss_type=loss_type,
                    logits=logits_post,
                    horizons_years=[float(business_horizon)],
                    criterion=criterion,
                )[0, 0]
            )
            post_risks.append(risk_post)

    baseline_arr = np.asarray(baseline_risks, dtype=np.float64)
    post_arr = np.asarray(post_risks, dtype=np.float64)
    if baseline_arr.size == 0:
        arr = float("nan")
        rrr = float("nan")
    else:
        diff = baseline_arr - post_arr
        diff = diff[np.isfinite(diff)]
        arr = float(np.mean(diff)) if diff.size > 0 else float("nan")
        denom = np.clip(baseline_arr, 1e-12, None)
        rel = (baseline_arr - post_arr) / denom
        rel = rel[np.isfinite(rel)]
        rrr = float(np.mean(rel)) if rel.size > 0 else float("nan")

    # -------------------------
    # Save summary JSON
    # -------------------------
    summary: Dict[str, object] = {
        "c_index": float(c_index),
        "aucs": aucs,
        "business_horizon_years": float(business_horizon),
        "enrichment_factor": float(enrichment_factor),
        "intervention": {
            "n_top": int(len(baseline_risks)),
            "bmi_cont_idx": None if bmi_cont_idx is None else int(bmi_cont_idx),
            "smoking_cate_idx": None if smoking_cate_idx is None else int(smoking_cate_idx),
            "smoking_non_smoker_value": int(smoking_non_smoker_value),
            "avg_absolute_risk_reduction": float(arr),
            "avg_relative_risk_reduction": float(rrr),
        },
        "files": {
            "risk_stratification": os.path.basename(strat_path),
            "dca_values": os.path.basename(dca_path),
        },
    }

    with open(os.path.join(out_dir, "clinical_metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate clinical utility metrics")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to a run directory under runs/*")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override checkpoint path (default: best_model.pt in run_dir)",
    )
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--horizons", type=str, default="1,3,5,10",
                        help="Comma-separated horizon years")
    parser.add_argument("--seed", type=int, default=42)

    # Business-metric knobs (kept minimal)
    parser.add_argument("--business_horizon", type=float, default=5.0,
                        help="Horizon (years) used for enrichment/DCA/intervention")
    parser.add_argument("--dca_threshold_max", type=float, default=0.5)
    parser.add_argument("--dca_threshold_steps", type=int, default=51)
    parser.add_argument("--topk_intervention", type=int, default=100)

    # Feature indices for counterfactual; defaults are inferred when possible.
    parser.add_argument("--bmi_cont_idx", type=int, default=None,
                        help="Continuous-feature index for BMI (in cont_seq[..., idx])")
    parser.add_argument("--smoking_cate_idx", type=int, default=None,
                        help="Categorical-feature index for smoking (in cate_seq[..., idx])")
    parser.add_argument("--smoking_non_smoker_value", type=int,
                        default=1, help="Categorical value representing 'Non-smoker'")

    args = parser.parse_args()

    run_dir = args.run_dir
    print("[Step] Loading training config...")
    train_cfg = _load_train_config(run_dir)

    if args.checkpoint is not None:
        checkpoint = args.checkpoint
    else:
        candidate = os.path.join(run_dir, "best_model.pt")
        fallback = os.path.join(run_dir, "best_checkpoint.pt")
        checkpoint = candidate if os.path.exists(candidate) else fallback

    out_dir = run_dir
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"[Step] Run dir: {run_dir}")
    print(f"[Step] Checkpoint: {checkpoint}")
    print(f"[Step] Output dir: {out_dir}")
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
    _train_data, _val_data, test_data = random_split(
        dataset, [n_train, n_val, n_test], generator=gen)

    print(
        f"[Step] Split sizes: train={n_train}, val={n_val}, test={len(test_data)}")

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

    horizons = tuple(float(x)
                     for x in str(args.horizons).split(",") if x.strip() != "")
    print(f"[Step] Horizons: {horizons}")
    print(f"[Step] Business horizon: {float(args.business_horizon)}y")

    # Infer feature indices if user didn't provide them.
    bmi_idx = args.bmi_cont_idx
    smoking_idx = args.smoking_cate_idx
    if bmi_idx is None or smoking_idx is None:
        inf_bmi, inf_smoking = _infer_feature_indices(
            data_prefix=data_prefix, covariate_list=cov_list)
        if bmi_idx is None:
            bmi_idx = inf_bmi
        if smoking_idx is None:
            smoking_idx = inf_smoking
    print(f"[Step] BMI cont idx: {bmi_idx}")
    print(f"[Step] Smoking cate idx: {smoking_idx}")
    print(
        f"[Step] Smoking non-smoker value: {int(args.smoking_non_smoker_value)}")

    summary = compute_clinical_metrics(
        model=model,
        criterion=criterion,
        loader=loader,
        loss_type=str(train_cfg["loss_type"]),
        horizons=horizons,
        out_dir=out_dir,
        seed=int(args.seed),
        business_horizon=float(args.business_horizon),
        dca_threshold_max=float(args.dca_threshold_max),
        dca_threshold_steps=int(args.dca_threshold_steps),
        topk_intervention=int(args.topk_intervention),
        bmi_cont_idx=bmi_idx,
        smoking_cate_idx=smoking_idx,
        smoking_non_smoker_value=int(args.smoking_non_smoker_value),
    )

    print("[Done] Clinical metrics computed. Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
