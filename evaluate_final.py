"""
Refactored evaluation for Delphi-style EHR survival models.

Key principles
- Landmark-aligned evaluation: scores are computed for time since landmark t0, not since last observed event.
- Prevent temporal misalignment: we insert an explicit landmark token at time t0 and use its hidden state for prediction.
- Incident disease evaluation: per-disease AUC excludes prevalent cases (disease observed at or before t0).
- Ranking evaluation: Recall@K / Precision@K on incident diseases within horizon.
- Competing-risk-consistent scoring:
    * exponential: closed-form competing risks in an interval
    * lognormal basis hazard: numerical quadrature for cause-specific probability mass in interval
"""

from __future__ import annotations
from dataset import HealthDataset, health_collate_fn
from model import DelphiFork, SapDelphi
from losses import LogNormalBasisHazardLoss, ExponentialNLLLoss, _gauss_legendre_16

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

remember_cwd = os.getcwd()
sys.path.append(remember_cwd)


# =============================================================================
# Utilities
# =============================================================================

DAYS_PER_YEAR = 365.25
PAD_EVENT = 0
LANDMARK_EVENT = 1  # DOA / landmark token id in this codebase


def load_labels(labels_file: str = "labels.csv") -> Dict[int, str]:
    """Load mapping from Disease ID to Name (token id -> string)."""
    labels_map: Dict[int, str] = {}
    if not os.path.exists(labels_file):
        return labels_map
    with open(labels_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            s = line.strip()
            if s:
                labels_map[idx + 2] = s
    return labels_map


def get_chapter(code_str: str) -> str:
    """Map disease name to ICD-10 Chapter (heuristic based on first letter)."""
    if not code_str:
        return "Unknown"
    letter = code_str[0].upper()
    mapping = {
        "A": "I: Infectious",
        "B": "I: Infectious",
        "C": "II: Neoplasms",
        "D": "III: Blood/Immune",
        "E": "IV: Metabolic",
        "F": "V: Mental",
        "G": "VI: Nervous",
        "H": "VII/VIII: Eye/Ear",
        "I": "IX: Circulatory",
        "J": "X: Respiratory",
        "K": "XI: Digestive",
        "L": "XII: Skin",
        "M": "XIII: Musculoskeletal",
        "N": "XIV: Genitourinary",
        "O": "XV: Pregnancy",
        "P": "XVI: Perinatal",
        "Q": "XVII: Congenital",
        "R": "XVIII: Symptoms",
        "S": "XIX: Injury",
        "T": "XIX: Injury",
        "Z": "XXI: Factors",
    }
    if "Death" in code_str or "death" in code_str:
        return "Death"
    return mapping.get(letter, "Other")


def _to_batch_time(t: float | int | torch.Tensor, B: int, device, dtype) -> torch.Tensor:
    if isinstance(t, (float, int)):
        return torch.full((B,), float(t), device=device, dtype=dtype)
    t = t.to(device=device, dtype=dtype)
    if t.ndim == 0:
        t = t.expand(B)
    return t


# =============================================================================
# Competing-risk-consistent risk scoring
# =============================================================================

def calculate_risk_score(
    theta: torch.Tensor,
    t_start: float | torch.Tensor,
    t_end: float | torch.Tensor,
    loss_type: str,
    loss_fn: Optional[LogNormalBasisHazardLoss] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Competing-risk-consistent cause-specific probability mass in (t_start, t_end]
    conditional on survival past t_start (i.e., landmark aligned).

    Inputs:
      - theta: (B, K, 1) for exponential or (B, K, n_basis) for lognormal
      - t_start, t_end: time-since-landmark in years (scalar or (B,))
    Output:
      - scores: (B, K), with sum over K <= 1 (up to numerical error)
    """
    device = theta.device
    dtype = theta.dtype
    B = theta.shape[0]

    t_start = _to_batch_time(t_start, B, device, dtype)
    t_end = _to_batch_time(t_end, B, device, dtype)

    dt = torch.clamp(t_end - t_start, min=0.0)  # (B,)

    if loss_type == "exponential":
        logits = theta
        if logits.dim() == 3 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)  # (B, K)
        lambdas = F.softplus(logits)  # (B, K)
        lam_tot = lambdas.sum(dim=1, keepdim=True).clamp_min(eps)  # (B,1)

        p_any = 1.0 - torch.exp(-lam_tot * dt.unsqueeze(1))  # (B,1)
        p_k = (lambdas / lam_tot) * p_any  # (B,K)
        return torch.clamp(p_k, 0.0, 1.0)

    if loss_type == "lognormal":
        if loss_fn is None:
            raise ValueError("loss_fn required for lognormal scoring")
        coeffs = theta  # (B, K, n_basis)

        # Quadrature nodes on [-1, 1]
        x_nodes, w = _gauss_legendre_16(
            device=device, dtype=dtype)  # (16,), (16,)

        # Helper: hazard at arbitrary u (years), for all diseases
        def hazards_at(u_years: torch.Tensor) -> torch.Tensor:
            """
            u_years: (B, Q) or (B*Q,)
            returns: (B, K, Q)
            """
            if u_years.ndim == 1:
                u_flat = u_years
                BQ = u_flat.shape[0]
                K_u_flat = loss_fn._compute_kernel(
                    torch.clamp(u_flat, min=1e-5))
                # infer Q
                Q = BQ // B
                K_u = K_u_flat.view(B, Q, -1)  # (B,Q,n_basis)
            else:
                B_, Q = u_years.shape
                assert B_ == B
                u_flat = u_years.reshape(-1)
                K_u_flat = loss_fn._compute_kernel(
                    torch.clamp(u_flat, min=1e-5))
                K_u = K_u_flat.view(B, Q, -1)

            log_h = torch.einsum("bkb,bqb->bkq", coeffs, K_u)  # (B,K,Q)
            log_h = torch.clamp(log_h, max=20.0)
            return torch.exp(log_h)  # (B,K,Q)

        # We need conditional survival over (t_start, t_end]:
        # S(t | t_start) = exp(-∫_{t_start}^{t} λ_tot(u) du).
        # We'll approximate the integral with ordered quadrature nodes.

        a = torch.clamp(t_start, min=1e-5)
        b = torch.clamp(t_end, min=1e-5)

        # Map nodes to [a,b] per-sample: u = (b-a)/2 * x + (a+b)/2
        u = ((b - a).unsqueeze(1) / 2.0) * x_nodes.unsqueeze(0) + \
            ((a + b).unsqueeze(1) / 2.0)  # (B,16)
        wu = ((b - a).unsqueeze(1) / 2.0) * w.unsqueeze(0)  # (B,16)

        h = hazards_at(u)  # (B,K,16)
        h_tot = h.sum(dim=1)  # (B,16)

        # Build conditional survival at each node using cumulative integral from a to u_node.
        # Sort nodes by time to create a monotone cumulative hazard.
        u_sorted, idx = torch.sort(u, dim=1)  # (B,16)
        wu_sorted = torch.gather(wu, 1, idx)  # (B,16)
        h_tot_sorted = torch.gather(h_tot, 1, idx)  # (B,16)

        # cumulative hazard increments (quadrature-weighted); this is a pragmatic approximation
        cum_H = torch.cumsum(h_tot_sorted * wu_sorted, dim=1)  # (B,16)

        inv_idx = torch.argsort(idx, dim=1)
        dH = torch.gather(cum_H, 1, inv_idx)  # (B,16)

        S_cond = torch.exp(-dH).clamp_min(0.0)  # (B,16)

        # Cause-specific probability mass in (a,b]: ∫ S_cond(t) * λ_k(t) dt
        Pk = torch.sum(h * S_cond.unsqueeze(1) *
                       wu.unsqueeze(1), dim=2)  # (B,K)
        return torch.clamp(Pk, 0.0, 1.0)

    raise ValueError(f"Unknown loss_type: {loss_type}")


# =============================================================================
# Landmark-aligned batch construction
# =============================================================================

@dataclass
class LandmarkBatch:
    e_in: torch.Tensor          # (B, L)
    t_in: torch.Tensor          # (B, L)
    cont: torch.Tensor          # (B, n_cont)
    cate: torch.Tensor          # (B, n_cate)
    sex: torch.Tensor           # (B,)
    targets: np.ndarray         # (B, K) float {0,1}
    prev: np.ndarray            # (B, K) bool
    valid_mask: np.ndarray      # (B,) bool (kept rows)


def _build_landmark_batch(
    event_batch: torch.Tensor,
    time_batch: torch.Tensor,
    cont_batch: torch.Tensor,
    cate_batch: torch.Tensor,
    sex_batch: torch.Tensor,
    *,
    base_dataset: HealthDataset,
    t0_years: float,
    buffer_years: float,
    horizon_years: float,
) -> LandmarkBatch:
    """
    Build landmark-aligned sequences:
      - truncate history to <= t0
      - append a landmark token at exactly t0
      - targets: incident disease occurrence in (t0+buffer, t0+horizon]
      - prev: disease occurs at or before t0
    """
    t0_days = t0_years * DAYS_PER_YEAR
    start_days = (t0_years + buffer_years) * DAYS_PER_YEAR
    end_days = (t0_years + horizon_years) * DAYS_PER_YEAR

    new_e, new_t, idx_kept = [], [], []
    targets, prevs = [], []

    B0 = event_batch.shape[0]

    for i in range(B0):
        valid = int((event_batch[i] != PAD_EVENT).sum().item())
        if valid <= 0:
            continue
        e_seq = event_batch[i, :valid]
        t_seq = time_batch[i, :valid]

        # Truncate to <= t0
        mask_hist = t_seq <= t0_days
        e_hist = e_seq[mask_hist]
        t_hist = t_seq[mask_hist]

        if t_hist.numel() == 0:
            # no history before landmark -> skip (or keep with only landmark)
            # keeping is OK, but some datasets may be empty. We'll keep with landmark only.
            e_hist = torch.tensor([LANDMARK_EVENT], dtype=e_seq.dtype)
            t_hist = torch.tensor([t0_days], dtype=t_seq.dtype)
        else:
            # Append explicit landmark token at t0 (even if last event is at t0)
            e_hist = torch.cat([e_hist, torch.tensor(
                [LANDMARK_EVENT], dtype=e_seq.dtype)])
            t_hist = torch.cat(
                [t_hist, torch.tensor([t0_days], dtype=t_seq.dtype)])

        # Prevalence up to t0 (exclude tech tokens <2)
        prev_vec = np.zeros(base_dataset.n_disease, dtype=bool)
        for c in e_seq[t_seq <= t0_days].tolist():
            if c >= 2:
                prev_vec[c - 2] = True

        # Incident targets in (t0+buffer, t0+horizon]
        tgt_vec = np.zeros(base_dataset.n_disease, dtype=np.float32)
        mask_tg = (t_seq > start_days) & (t_seq <= end_days)
        for c in e_seq[mask_tg].tolist():
            if c >= 2:
                tgt_vec[c - 2] = 1.0

        new_e.append(e_hist)
        new_t.append(t_hist)
        idx_kept.append(i)
        targets.append(tgt_vec)
        prevs.append(prev_vec)

    if len(new_e) == 0:
        # Return empty placeholders
        return LandmarkBatch(
            e_in=torch.empty((0, 1), dtype=event_batch.dtype,
                             device=event_batch.device),
            t_in=torch.empty((0, 1), dtype=time_batch.dtype,
                             device=time_batch.device),
            cont=torch.empty(
                (0,) + cont_batch.shape[1:], dtype=cont_batch.dtype, device=cont_batch.device),
            cate=torch.empty(
                (0,) + cate_batch.shape[1:], dtype=cate_batch.dtype, device=cate_batch.device),
            sex=torch.empty((0,), dtype=sex_batch.dtype,
                            device=sex_batch.device),
            targets=np.zeros((0, base_dataset.n_disease), dtype=np.float32),
            prev=np.zeros((0, base_dataset.n_disease), dtype=bool),
            valid_mask=np.zeros((B0,), dtype=bool),
        )

    from torch.nn.utils.rnn import pad_sequence
    e_in = pad_sequence(new_e, batch_first=True).to(event_batch.device)
    t_in = pad_sequence(new_t, batch_first=True,
                        padding_value=100 * DAYS_PER_YEAR).to(time_batch.device)

    idx_kept_t = torch.tensor(idx_kept, device=cont_batch.device)
    cont = cont_batch[idx_kept_t]
    cate = cate_batch[idx_kept_t]
    sex = sex_batch[idx_kept_t]

    valid_mask = np.zeros((B0,), dtype=bool)
    valid_mask[idx_kept] = True

    return LandmarkBatch(
        e_in=e_in,
        t_in=t_in,
        cont=cont,
        cate=cate,
        sex=sex,
        targets=np.asarray(targets),
        prev=np.asarray(prevs),
        valid_mask=valid_mask,
    )


def _predict_theta_at_landmark(
    model,
    batch: LandmarkBatch,
    *,
    n_disease: int,
) -> torch.Tensor:
    """
    Run model and extract theta at landmark token position (last token in each sequence).
    Returns theta shaped (B, K, n_dim) or (B, K, 1) depending on model.
    """
    if batch.e_in.shape[0] == 0:
        return torch.empty((0, n_disease, 1), device=batch.e_in.device)

    B = batch.e_in.shape[0]
    # landmark token is appended as last token
    b_prev = torch.arange(B, device=batch.e_in.device)
    t_prev = torch.tensor([batch.e_in.shape[1] - 1] *
                          B, device=batch.e_in.device)

    theta = model(batch.e_in, batch.t_in, batch.sex, batch.cont,
                  batch.cate, b_prev=b_prev, t_prev=t_prev)

    # Normalize theta to (B, K, n_dim)
    if theta.dim() == 2:
        # (B, K*n_dim)
        theta = theta.reshape(B, n_disease, -1)
    elif theta.dim() == 3:
        # Either (B, K, n_dim) or (B, ?, ?). Assume (B, K, n_dim).
        if theta.shape[1] != n_disease:
            # last resort: reshape
            theta = theta.reshape(B, n_disease, -1)
    else:
        raise ValueError(f"Unexpected theta shape: {theta.shape}")

    return theta


# =============================================================================
# Evaluation: landmark-aligned AUC and ranking
# =============================================================================

def _compute_macro_auc(risk: np.ndarray, targets: np.ndarray, prev: np.ndarray, min_pos: int = 5) -> float:
    """
    Macro-average per-disease ROC-AUC over non-prevalent individuals.
    """
    K = targets.shape[1]
    aucs: List[float] = []
    for k in range(K):
        mask = ~prev[:, k]
        y = targets[mask, k]
        if y.size < 10 or int(y.sum()) < min_pos:
            continue
        try:
            aucs.append(roc_auc_score(y, risk[mask, k]))
        except Exception:
            continue
    return float(np.mean(aucs)) if aucs else 0.0


def _compute_weighted_auc(risk: np.ndarray, targets: np.ndarray, prev: np.ndarray, min_pos: int = 5) -> float:
    """
    Weighted macro AUC: weight each disease by its positive count among non-prevalent.
    """
    K = targets.shape[1]
    aucs, weights = [], []
    for k in range(K):
        mask = ~prev[:, k]
        y = targets[mask, k]
        pos = int(y.sum())
        if y.size < 10 or pos < min_pos:
            continue
        try:
            aucs.append(roc_auc_score(y, risk[mask, k]))
            weights.append(pos)
        except Exception:
            continue
    if not aucs:
        return 0.0
    w = np.asarray(weights, dtype=np.float64)
    w = w / (w.sum() + 1e-12)
    return float((np.asarray(aucs) * w).sum())


def _compute_recall_precision_at_k(
    risk: np.ndarray,
    targets: np.ndarray,
    prev: np.ndarray,
    ks: Tuple[int, ...] = (10, 20),
) -> Dict[str, float]:
    """
    Ranking metrics on incident diseases:
      - mask prevalent diseases by setting their score to -inf
      - compute per-person Recall@K and Precision@K, then average over persons with >=1 target.
    """
    scores = risk.copy()
    scores[prev] = -np.inf

    res: Dict[str, float] = {}
    for k in ks:
        recalls, precisions = [], []
        for i in range(scores.shape[0]):
            true_idx = np.where(targets[i] > 0)[0]
            if true_idx.size == 0:
                continue
            top = np.argsort(scores[i])[::-1][:k]
            hit = np.isin(top, true_idx).sum()
            recalls.append(hit / true_idx.size)
            precisions.append(hit / k)
        res[f"Recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        res[f"Precision@{k}"] = float(np.mean(precisions)
                                      ) if precisions else 0.0
    return res


def run_landmark_evaluation(
    dataset,
    model,
    loss_fn,
    args,
    *,
    ages: List[int],
    horizons: List[float],
    buffer_years: float = 0.5,
    min_N: int = 100,
) -> None:
    """
    Landmark evaluation across age buckets and horizons.
    Produces:
      - results_auc.csv (macro and weighted AUC)
      - results_ranking.csv (Recall@K / Precision@K)
    """
    print("\n" + "=" * 70)
    print(">>> Landmark-Aligned Evaluation (AUC + Ranking)")
    print("=" * 70)

    # Resolve base dataset for metadata access
    if isinstance(dataset, Subset):
        base_dataset: HealthDataset = dataset.dataset
        def get_real_idx(i): return dataset.indices[i]
    else:
        base_dataset = dataset
        def get_real_idx(i): return i

    # Precompute follow-up length and optional death time from raw event list
    disease_map = load_labels()
    death_id = None
    for k, v in disease_map.items():
        if "Death" in v or "death" in v:
            death_id = k
            break

    patient_meta = []
    for i in range(len(dataset)):
        pid = base_dataset.patient_ids[get_real_idx(i)]
        events = base_dataset.patient_events[pid]
        if not events:
            continue
        events = sorted(events, key=lambda x: x[0])
        max_t = events[-1][0] / DAYS_PER_YEAR
        d_t = None
        if death_id is not None:
            for t, c in events:
                if c == death_id:
                    d_t = t / DAYS_PER_YEAR
                    break
        patient_meta.append({"idx": i, "max_t": max_t, "d_t": d_t})

    auc_rows = []
    rank_rows = []

    for age in ages:
        t0 = float(age)

        # For eligibility, require that outcome window is at least potentially observable.
        # We keep people who either (a) have follow-up past t0 + max(horizon),
        # or (b) die before that (then outcomes after death are impossible; still informative for AUC as negatives).
        max_h = max(horizons)
        t_end_req = t0 + max_h

        eligible = []
        for pm in patient_meta:
            if pm["d_t"] is not None and pm["d_t"] <= t0:
                continue
            if pm["max_t"] >= t_end_req or (pm["d_t"] is not None and pm["d_t"] <= t_end_req):
                eligible.append(pm["idx"])

        if len(eligible) < min_N:
            print(f"Age {age}: skipped (N={len(eligible)})")
            continue

        loader = DataLoader(
            Subset(dataset, eligible),
            batch_size=args.batch_size,
            collate_fn=health_collate_fn,
            shuffle=False,
        )

        for h in horizons:
            all_risk, all_targets, all_prev = [], [], []

            with torch.no_grad():
                for batch in tqdm(loader, desc=f"Age {age} | H {h}", leave=False):
                    event_batch, time_batch, cont_batch, cate_batch, sex_batch = batch
                    # Move to device early (collate may already be on CPU)
                    event_batch = event_batch.to(args.device)
                    time_batch = time_batch.to(args.device)
                    cont_batch = cont_batch.to(args.device)
                    cate_batch = cate_batch.to(args.device)
                    sex_batch = sex_batch.to(args.device)

                    lm = _build_landmark_batch(
                        event_batch, time_batch, cont_batch, cate_batch, sex_batch,
                        base_dataset=base_dataset,
                        t0_years=t0,
                        buffer_years=buffer_years,
                        horizon_years=h,
                    )

                    if lm.e_in.shape[0] == 0:
                        continue

                    theta = _predict_theta_at_landmark(
                        model, lm, n_disease=base_dataset.n_disease)

                    # Risk score for interval (buffer, h] since landmark
                    risk_t = calculate_risk_score(
                        theta, buffer_years, float(h), args.loss_type, loss_fn)

                    all_risk.append(risk_t.detach().cpu().numpy())
                    all_targets.append(lm.targets)
                    all_prev.append(lm.prev)

            if not all_risk:
                continue

            risk = np.concatenate(all_risk, axis=0)
            targets = np.concatenate(all_targets, axis=0)
            prev = np.concatenate(all_prev, axis=0)

            macro_auc = _compute_macro_auc(risk, targets, prev)
            w_auc = _compute_weighted_auc(risk, targets, prev)

            rank = _compute_recall_precision_at_k(
                risk, targets, prev, ks=(10, 20))

            print(
                f"Age {age:>2} | H {h:>4}y | macroAUC {macro_auc:.4f} | wAUC {w_auc:.4f} | "
                f"R@10 {rank['Recall@10']:.4f} | R@20 {rank['Recall@20']:.4f}"
            )

            auc_rows.append({
                "Age": age,
                "HorizonYears": h,
                "BufferYears": buffer_years,
                "N": int(risk.shape[0]),
                "MacroAUC": macro_auc,
                "WeightedAUC": w_auc,
            })
            rank_rows.append({
                "Age": age,
                "HorizonYears": h,
                "BufferYears": buffer_years,
                "N": int(risk.shape[0]),
                **rank,
            })

    os.makedirs(args.run_dir, exist_ok=True)
    pd.DataFrame(auc_rows).to_csv(os.path.join(
        args.run_dir, "results_auc.csv"), index=False)
    pd.DataFrame(rank_rows).to_csv(os.path.join(
        args.run_dir, "results_ranking.csv"), index=False)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--loss_type", type=str, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    cfg_path = os.path.join(args.run_dir, "train_config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    args.loss_type = args.loss_type or cfg.get("loss_type", "lognormal")

    if args.loss_type == "exponential":
        loss_fn = ExponentialNLLLoss()
        n_dim = 1
    elif args.loss_type == "lognormal":
        bin_edges = cfg.get("bin_edges", (0.01, 0.09, 0.23,
                            0.44, 0.72, 1.07, 1.61, 2.4, 3.84, 7.0, 31.0))
        loss_fn = LogNormalBasisHazardLoss(centers=list(bin_edges))
        n_dim = len(bin_edges)
    else:
        raise ValueError(f"Unknown loss_type: {args.loss_type}")

    loss_fn.to(args.device)

    data_prefix = cfg.get("data_prefix", "ukb")
    cov = None if cfg.get("full_cov") else ["bmi", "smoking", "alcohol"]
    dataset = HealthDataset(data_prefix=data_prefix, covariate_list=cov)

    model_type = cfg.get("model_type", "delphifork")
    n_embd = cfg.get("n_embd", 120)

    if model_type == "delphifork":
        model = DelphiFork(
            n_disease=dataset.n_disease,
            n_tech_tokens=2,
            n_cont=dataset.n_cont,
            n_cate=dataset.n_cate,
            cate_dims=dataset.cate_dims,
            n_embd=n_embd,
            n_layer=cfg.get("n_layer", 12),
            n_head=cfg.get("n_head", 12),
            pdrop=0.0,
            age_encoder_type=cfg.get("age_encoder", "sinusoidal"),
            n_dim=n_dim,
        )
    elif model_type == "sapdelphi":
        model = SapDelphi(
            n_disease=dataset.n_disease,
            n_tech_tokens=2,
            n_cont=dataset.n_cont,
            n_cate=dataset.n_cate,
            cate_dims=dataset.cate_dims,
            n_embd=n_embd,
            n_layer=cfg.get("n_layer", 12),
            n_head=cfg.get("n_head", 12),
            pdrop=0.0,
            age_encoder_type=cfg.get("age_encoder", "sinusoidal"),
            n_dim=n_dim,
            pretrained_weights_path=cfg.get("pretrained_weights_path"),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.to(args.device)

    ckpt_path = os.path.join(args.run_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args.run_dir, "last_model.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(state["model_state_dict"])
    if "criterion_state_dict" in state and loss_fn is not None:
        loss_fn.load_state_dict(state["criterion_state_dict"])
    model.eval()

    # Test split (same as training script)
    n_total = len(dataset)
    tr = int(n_total * cfg.get("train_ratio", 0.7))
    va = int(n_total * cfg.get("val_ratio", 0.15))
    te = n_total - tr - va
    _, _, test_dataset = random_split(
        dataset,
        [tr, va, te],
        generator=torch.Generator().manual_seed(cfg.get("random_seed", 42)),
    )

    ages = cfg.get("eval_ages", [40, 45, 50, 55, 60, 65, 70, 75, 80])
    horizons = cfg.get("eval_horizons", [1.0, 5.0, 10.0])
    buffer_years = float(cfg.get("eval_buffer_years", 0.5))

    run_landmark_evaluation(
        test_dataset,
        model,
        loss_fn,
        args,
        ages=list(ages),
        horizons=list(horizons),
        buffer_years=buffer_years,
        min_N=int(cfg.get("eval_min_N", 100)),
    )

    print("Done.")


if __name__ == "__main__":
    main()
