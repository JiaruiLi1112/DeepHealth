from losses import LogNormalBasisHazardLoss, ExponentialNLLLoss, _gauss_legendre_16
from model import DelphiFork, SapDelphi
from dataset import HealthDataset, health_collate_fn
import os
import sys
import json
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm import tqdm

# Ensure local modules can be imported
sys.path.append(os.getcwd())


# =============================================================================
# Helper Functions
# =============================================================================


def load_labels(labels_file="labels.csv"):
    """Load mapping from Disease ID to Name."""
    labels_map = {}
    if not os.path.exists(labels_file):
        return labels_map
    with open(labels_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            parts = line.strip()
            if parts:
                labels_map[idx + 2] = parts
    return labels_map


def get_chapter(code_str):
    """Map disease name to ICD-10 Chapter."""
    if not code_str:
        return "Unknown"
    letter = code_str[0].upper()
    mapping = {
        'A': "I: Infectious", 'B': "I: Infectious",
        'C': "II: Neoplasms", 'D': "III: Blood/Immune",
        'E': "IV: Metabolic", 'F': "V: Mental",
        'G': "VI: Nervous", 'H': "VII/VIII: Eye/Ear",
        'I': "IX: Circulatory", 'J': "X: Respiratory",
        'K': "XI: Digestive", 'L': "XII: Skin",
        'M': "XIII: Musculoskeletal", 'N': "XIV: Genitourinary",
        'O': "XV: Pregnancy", 'P': "XVI: Perinatal",
        'Q': "XVII: Congenital", 'R': "XVIII: Symptoms",
        'S': "XIX: Injury", 'T': "XIX: Injury",
        'Z': "XXI: Factors"
    }
    if "Death" in code_str or "death" in code_str:
        return "Death"
    return mapping.get(letter, "Other")


def calculate_risk_score(theta, t_start, t_end, loss_type, loss_fn=None):
    device = theta.device

    if loss_type == "exponential":
        logits = theta
        if logits.dim() == 3:
            logits = logits.squeeze(-1)
        lambdas = F.softplus(logits)  # (B, K)

        dt = t_end - t_start
        if isinstance(dt, torch.Tensor) and dt.ndim == 1:
            dt = dt.unsqueeze(-1)  # (B, 1)

        lambda_total = torch.sum(lambdas, dim=1, keepdim=True)  # (B, 1)
        lambda_total_safe = torch.clamp(lambda_total, min=1e-12)

        total_event_prob = 1.0 - torch.exp(-lambda_total_safe * dt)
        scores = (lambdas / lambda_total_safe) * total_event_prob

        scores = torch.clamp(scores, min=0.0)
        cap = torch.clamp(total_event_prob, min=0.0, max=1.0)
        total = torch.sum(scores, dim=1, keepdim=True)
        scale = torch.where(
            total > cap, cap / torch.clamp(total, min=1e-12), torch.ones_like(total))
        scores = scores * scale

        return scores

    if loss_type == "lognormal":
        if loss_fn is None:
            raise ValueError("loss_fn required for lognormal")

        coeffs = theta
        if coeffs.dim() != 3:
            raise ValueError("lognormal expects theta shaped (B, K, n_basis)")

        B, K, n_basis = coeffs.shape

        def _as_B_vector(v):
            if isinstance(v, (float, int)):
                return torch.full((B,), float(v), device=device, dtype=coeffs.dtype)
            if isinstance(v, torch.Tensor):
                if v.ndim == 0:
                    return torch.full((B,), float(v.item()), device=device, dtype=coeffs.dtype)
                if v.ndim == 1:
                    return v.to(device=device, dtype=coeffs.dtype)
                raise ValueError("t_start/t_end must be scalar or (B,) tensor")
            raise ValueError("t_start/t_end must be scalar or tensor")

        t_start_vec = _as_B_vector(t_start)
        t_end_vec = _as_B_vector(t_end)

        t_start_vec = torch.clamp(t_start_vec, min=1e-5)
        t_end_vec = torch.clamp(t_end_vec, min=1e-5)
        dt = torch.clamp(t_end_vec - t_start_vec, min=0.0)

        x_nodes, w_nodes = _gauss_legendre_16(device=device, dtype=coeffs.dtype)

        def hazards_at_times(t_vals_bm: torch.Tensor) -> torch.Tensor:
            t = torch.clamp(t_vals_bm, min=1e-5)
            t_flat = t.reshape(-1)
            K_flat = loss_fn._compute_kernel(t_flat)  # (B*M, n_basis)
            K_bm = K_flat.view(t.shape[0], t.shape[1], -1)  # (B, M, n_basis)

            log_h = torch.einsum("bkb,bmb->bkm", coeffs, K_bm)
            log_h = torch.clamp(log_h, max=20.0)
            return torch.exp(log_h)  # (B, K, M)

        def H_total(t_vals_bm: torch.Tensor) -> torch.Tensor:
            t = torch.clamp(t_vals_bm, min=1e-5)
            B_local, M_local = t.shape

            u = (t.unsqueeze(-1) / 2.0) * \
                (x_nodes.view(1, 1, -1) + 1.0)  # (B, M, Q)
            weights = (t.unsqueeze(-1) / 2.0) * \
                w_nodes.view(1, 1, -1)  # (B, M, Q)

            u_flat = u.reshape(-1)
            K_flat = loss_fn._compute_kernel(u_flat)  # (B*M*Q, n_basis)
            # (B, M, Q, n_basis)
            K_bmqb = K_flat.view(B_local, M_local, u.shape[-1], -1)

            log_h = torch.einsum("bkb,bmqb->bkmq", coeffs, K_bmqb)
            log_h = torch.clamp(log_h, max=20.0)
            hazards = torch.exp(log_h)  # (B, K, M, Q)

            lambda_total = torch.sum(hazards, dim=1)  # (B, M, Q)
            H = torch.sum(lambda_total * weights, dim=-1)  # (B, M)
            return H

        Q = x_nodes.numel()
        dt_half = dt / 2.0
        u_outer = t_start_vec.unsqueeze(-1) + dt_half.unsqueeze(-1) * \
            (x_nodes.view(1, -1) + 1.0)  # (B, Q)
        w_outer = dt_half.unsqueeze(-1) * w_nodes.view(1, -1)  # (B, Q)

        hazards_k_u = hazards_at_times(u_outer)  # (B, K, Q)

        H_total_start = H_total(t_start_vec.view(B, 1)).squeeze(1)  # (B,)
        H_total_end = H_total(t_end_vec.view(B, 1)).squeeze(1)  # (B,)
        H_total_u = H_total(u_outer)  # (B, Q)

        delta_H_u = torch.clamp(
            H_total_u - H_total_start.unsqueeze(-1), min=0.0)
        S_u = torch.exp(-torch.clamp(delta_H_u, max=60.0))  # (B, Q)

        integrand = hazards_k_u * S_u.unsqueeze(1)  # (B, K, Q)
        scores = torch.sum(integrand * w_outer.unsqueeze(1), dim=-1)  # (B, K)
        scores = torch.clamp(scores, min=0.0)

        delta_H_end = torch.clamp(H_total_end - H_total_start, min=0.0)
        cap = torch.clamp(1.0 - torch.exp(-torch.clamp(delta_H_end,
                          max=60.0)), min=0.0, max=1.0).unsqueeze(-1)
        total = torch.sum(scores, dim=1, keepdim=True)
        scale = torch.where(
            total > cap, cap / torch.clamp(total, min=1e-12), torch.ones_like(total))
        scores = scores * scale

        return scores

    raise ValueError(f"Unknown loss_type: {loss_type}")

# =============================================================================
# Experiment 1: Age-Stratified Evaluation (Dual Metrics)
# =============================================================================


def run_stratified_evaluation(dataset, model, loss_fn, args, disease_map):
    print("\n" + "="*60)
    print(">>> Experiment 1: Age-Stratified Evaluation (Dual Metrics)")
    print("="*60)

    # Setup Metadata
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        def get_real_idx(i): return dataset.indices[i]
    else:
        base_dataset = dataset
        def get_real_idx(i): return i

    print("Scanning metadata...")
    patient_meta = []
    death_id = None
    for k, v in disease_map.items():
        if "Death" in v:
            death_id = k
            break

    for i in range(len(dataset)):
        pid = base_dataset.patient_ids[get_real_idx(i)]
        events = base_dataset.patient_events[pid]
        if not events:
            continue
        events = sorted(events, key=lambda x: x[0])
        max_t = events[-1][0]/365.25
        d_t = None
        if death_id:
            for t, c in events:
                if c == death_id:
                    d_t = t/365.25
                    break
        patient_meta.append({'idx': i, 'max_t': max_t, 'd_t': d_t})

    age_buckets = [40, 45, 50, 55, 60, 65, 70, 75, 80]
    results = []

    for age in age_buckets:
        t_start = float(age)
        t_window = 5.0
        t_end = t_start + t_window

        # Strict Filtering
        eligible_indices = []
        for pm in patient_meta:
            if pm['d_t'] is not None and pm['d_t'] <= t_start:
                continue
            if pm['max_t'] >= t_end or (pm['d_t'] is not None and pm['d_t'] <= t_end):
                eligible_indices.append(pm['idx'])

        if len(eligible_indices) < 50:
            print(f"Age {age}: Skipped (N={len(eligible_indices)})")
            continue

        loader = DataLoader(
            Subset(dataset, eligible_indices),
            batch_size=args.batch_size, collate_fn=health_collate_fn, shuffle=False
        )

        # Storage for Dual Metrics
        all_risks_temp = []   # Temporal (With Gap)
        all_risks_static = []  # Static (No Gap)
        all_targets = []
        all_prev = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Age {age}", leave=False):
                event_batch, time_batch, cont_batch, cate_batch, sex_batch = batch

                limit_days = t_start * 365.25
                target_start_d = (t_start + 0.5) * 365.25
                target_end_d = (t_start + 5.0) * 365.25

                new_e, new_t, b_idx = [], [], []
                batch_tg, batch_pr = [], []

                for k in range(event_batch.shape[0]):
                    valid = (event_batch[k] != 0).sum().item()
                    e_seq = event_batch[k, :valid]
                    t_seq = time_batch[k, :valid]

                    mask_in = t_seq <= limit_days
                    t_trunc = t_seq[mask_in]
                    e_trunc = e_seq[mask_in]
                    if len(t_trunc) == 0:
                        continue

                    # Target
                    mask_tg = (t_seq > target_start_d) & (t_seq <= target_end_d)
                    tgt_codes = e_seq[mask_tg].tolist()
                    tgt_vec = np.zeros(base_dataset.n_disease)
                    for c in tgt_codes:
                        if c >= 2:
                            tgt_vec[c-2] = 1.0

                    # Prevalence
                    prev_vec = np.zeros(base_dataset.n_disease, dtype=bool)
                    for c in e_trunc:
                        if c >= 2:
                            prev_vec[c-2] = True

                    new_e.append(e_trunc)
                    new_t.append(t_trunc)
                    batch_tg.append(tgt_vec)
                    batch_pr.append(prev_vec)
                    b_idx.append(k)

                if not new_e:
                    continue

                from torch.nn.utils.rnn import pad_sequence
                e_in = pad_sequence(new_e, batch_first=True).to(args.device)
                t_in = pad_sequence(new_t, batch_first=True,
                                    padding_value=36525.0).to(args.device)

                k_idx = torch.tensor(b_idx, device=cont_batch.device)
                c_in = cont_batch[k_idx].to(args.device)
                ca_in = cate_batch[k_idx].to(args.device)
                s_in = sex_batch[k_idx].to(args.device)

                b_prev = torch.arange(len(new_e), device=args.device)
                t_prev = torch.tensor(
                    [len(x)-1 for x in new_e], device=args.device)

                theta = model(e_in, t_in, s_in, c_in, ca_in,
                              b_prev=b_prev, t_prev=t_prev)
                theta = theta.view(len(new_e), base_dataset.n_disease, -1)

                # --- 1. Temporal Risk (With Gap) ---
                last_times = torch.tensor([t[-1]
                                          for t in new_t], device=args.device)
                gap = t_start - (last_times / 365.25)
                gap = torch.clamp(gap, min=0.0)

                r_temp = calculate_risk_score(
                    theta, gap+0.5, gap+5.0, args.loss_type, loss_fn)
                all_risks_temp.append(r_temp.cpu().numpy())

                # --- 2. Static Risk (Memoryless) ---
                # Assume Gap = 0 for everyone. P(0.5 < T < 5.0)
                # This measures pure "severity"
                r_static = calculate_risk_score(
                    theta, 0.5, 5.0, args.loss_type, loss_fn)
                all_risks_static.append(r_static.cpu().numpy())

                all_targets.append(np.array(batch_tg))
                all_prev.append(np.array(batch_pr))

        if not all_risks_temp:
            continue

        # Aggregate
        r_temp_all = np.concatenate(all_risks_temp, axis=0)
        r_stat_all = np.concatenate(all_risks_static, axis=0)
        t_all = np.concatenate(all_targets, axis=0)
        p_all = np.concatenate(all_prev, axis=0)

        # Compute AUCs
        auc_temp_list, auc_stat_list = [], []

        for d_idx in range(base_dataset.n_disease):
            mask = ~p_all[:, d_idx]  # Remove prevalent cases
            y_true = t_all[mask, d_idx]

            if np.sum(y_true) < 5 or len(y_true) < 10:
                continue

            # Temporal AUC
            try:
                auc_t = roc_auc_score(y_true, r_temp_all[mask, d_idx])
                auc_temp_list.append(auc_t)
            except:
                pass

            # Static AUC
            try:
                auc_s = roc_auc_score(y_true, r_stat_all[mask, d_idx])
                auc_stat_list.append(auc_s)
            except:
                pass

        mean_temp = np.mean(auc_temp_list) if auc_temp_list else 0.0
        mean_stat = np.mean(auc_stat_list) if auc_stat_list else 0.0

        print(
            f"Age {age} | Temporal AUC: {mean_temp:.4f} | Static AUC: {mean_stat:.4f}")

        results.append({
            'Age': age,
            'Temporal_AUC': mean_temp,
            'Static_AUC': mean_stat,
            'N_Patients': len(eligible_indices)
        })

    pd.DataFrame(results).to_csv(os.path.join(
        args.run_dir, "results_exp1_stratified.csv"), index=False)


# =============================================================================
# Experiment 2: Landmark (Using Static/Memoryless for Ranking)
# =============================================================================

def run_landmark_analysis(dataset, model, loss_fn, args):
    print("\n" + "="*60)
    print(">>> Experiment 2: Landmark Analysis (Static Ranking)")
    print("="*60)

    # Simplified for robustness: Use Static Risk for Ranking
    t_landmark = 60.0
    limit_days = t_landmark * 365.25
    min_future_days = (t_landmark + 1.0) * 365.25

    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        def get_real_idx(i): return dataset.indices[i]
    else:
        base_dataset = dataset
        def get_real_idx(i): return i

    eligible_indices = []
    for i in range(len(dataset)):
        pid = base_dataset.patient_ids[get_real_idx(i)]
        events = base_dataset.patient_events[pid]
        if events and events[-1][0] > min_future_days:
            eligible_indices.append(i)

    if not eligible_indices:
        return

    loader = DataLoader(
        Subset(dataset, eligible_indices),
        batch_size=args.batch_size, collate_fn=health_collate_fn, shuffle=False
    )

    horizons = [5, 10, 20]
    res_rank = []

    collated_preds = {h: [] for h in horizons}
    collated_targets = {h: [] for h in horizons}
    collated_prev = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Landmark", leave=False):
            event_batch, time_batch, cont_batch, cate_batch, sex_batch = batch

            new_e, new_t, b_idx = [], [], []
            batch_tg = {h: [] for h in horizons}
            batch_pr = []

            for k in range(event_batch.shape[0]):
                valid = (event_batch[k] != 0).sum().item()
                e_seq = event_batch[k, :valid]
                t_seq = time_batch[k, :valid]

                mask_in = t_seq <= limit_days
                e_trunc = e_seq[mask_in]
                t_trunc = t_seq[mask_in]
                if len(t_trunc) == 0:
                    continue

                # Prevalence
                prev_vec = np.zeros(base_dataset.n_disease, dtype=bool)
                for c in e_trunc:
                    if c >= 2:
                        prev_vec[c-2] = True

                # Targets
                for h in horizons:
                    end_d = (t_landmark + h) * 365.25
                    mask_tg = (t_seq > min_future_days) & (t_seq <= end_d)
                    tgt = np.zeros(base_dataset.n_disease)
                    for c in e_seq[mask_tg]:
                        if c >= 2:
                            tgt[c-2] = 1.0
                    batch_tg[h].append(tgt)

                new_e.append(e_trunc)
                new_t.append(t_trunc)
                b_idx.append(k)
                batch_pr.append(prev_vec)

            if not new_e:
                continue

            from torch.nn.utils.rnn import pad_sequence
            e_in = pad_sequence(new_e, batch_first=True).to(args.device)
            t_in = pad_sequence(new_t, batch_first=True,
                                padding_value=36525.0).to(args.device)
            k_idx = torch.tensor(b_idx, device=cont_batch.device)
            c_in = cont_batch[k_idx].to(args.device)
            ca_in = cate_batch[k_idx].to(args.device)
            s_in = sex_batch[k_idx].to(args.device)

            b_prev = torch.arange(len(new_e), device=args.device)
            t_prev = torch.tensor([len(x)-1 for x in new_e], device=args.device)

            theta = model(e_in, t_in, s_in, c_in, ca_in,
                          b_prev=b_prev, t_prev=t_prev)
            theta = theta.view(len(new_e), base_dataset.n_disease, -1)

            collated_prev.append(np.array(batch_pr))

            for h in horizons:
                # Use Static Risk for Ranking (Robust)
                # This predicts general severity in next H years, ignoring Gap noise
                r = calculate_risk_score(
                    theta, 1.0, float(h), args.loss_type, loss_fn)
                collated_preds[h].append(r.cpu().numpy())
                collated_targets[h].append(np.array(batch_tg[h]))

    prev_all = np.concatenate(collated_prev, axis=0)

    for h in horizons:
        if not collated_preds[h]:
            continue
        preds = np.concatenate(collated_preds[h], axis=0)
        trues = np.concatenate(collated_targets[h], axis=0)

        # Mask Prevalence for Ranking
        preds[prev_all] = -1.0

        recalls, precisions = [], []
        for k in [10, 20]:
            r_k, p_k = [], []
            for i in range(len(preds)):
                t_idx = np.where(trues[i] > 0)[0]
                if len(t_idx) == 0:
                    continue

                sort_idx = np.argsort(preds[i])[::-1]
                top = sort_idx[:k]
                hit = np.isin(top, t_idx).sum()
                r_k.append(hit/len(t_idx))
                p_k.append(hit/k)

            recalls.append(np.mean(r_k))
            precisions.append(np.mean(p_k))

        print(
            f"Horizon {h} | Recall@10: {recalls[0]:.4f} | Recall@20: {recalls[1]:.4f}")
        res_rank.append({
            'Horizon': h,
            'Recall_10': recalls[0], 'Precision_10': precisions[0],
            'Recall_20': recalls[1], 'Precision_20': precisions[1]
        })

    pd.DataFrame(res_rank).to_csv(os.path.join(
        args.run_dir, "results_exp2_ranking.csv"), index=False)


if __name__ == "__main__":
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
    loss_fn.to(args.device)

    data_prefix = cfg.get("data_prefix", "ukb")
    cov = None if cfg.get("full_cov") else ["bmi", "smoking", "alcohol"]
    dataset = HealthDataset(data_prefix=data_prefix, covariate_list=cov)

    # Model Init (Standard)
    model_type = cfg.get("model_type", "delphifork")
    n_embd = cfg.get("n_embd", 120)

    if model_type == "delphifork":
        model = DelphiFork(
            n_disease=dataset.n_disease, n_tech_tokens=2, n_cont=dataset.n_cont,
            n_cate=dataset.n_cate, cate_dims=dataset.cate_dims, n_embd=n_embd,
            n_layer=cfg.get("n_layer", 12), n_head=cfg.get("n_head", 12),
            pdrop=0.0, age_encoder_type=cfg.get("age_encoder", "sinusoidal"), n_dim=n_dim
        )
    elif model_type == "sapdelphi":
        model = SapDelphi(
            n_disease=dataset.n_disease, n_tech_tokens=2, n_cont=dataset.n_cont,
            n_cate=dataset.n_cate, cate_dims=dataset.cate_dims, n_embd=n_embd,
            n_layer=cfg.get("n_layer", 12), n_head=cfg.get("n_head", 12),
            pdrop=0.0, age_encoder_type=cfg.get("age_encoder", "sinusoidal"), n_dim=n_dim,
            pretrained_weights_path=cfg.get("pretrained_weights_path")
        )

    model.to(args.device)

    ckpt_path = os.path.join(args.run_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args.run_dir, "last_model.pt")
    print(f"Loading: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(state['model_state_dict'])
    if 'criterion_state_dict' in state and loss_fn:
        loss_fn.load_state_dict(state['criterion_state_dict'])
    model.eval()

    # Split
    n_total = len(dataset)
    tr = int(n_total * cfg.get("train_ratio", 0.7))
    va = int(n_total * cfg.get("val_ratio", 0.15))
    te = n_total - tr - va
    _, _, test_dataset = random_split(
        dataset, [tr, va, te], generator=torch.Generator(
        ).manual_seed(cfg.get("random_seed", 42))
    )

    run_stratified_evaluation(test_dataset, model, loss_fn, args, load_labels())
    run_landmark_analysis(test_dataset, model, loss_fn, args)
    print("Done.")
