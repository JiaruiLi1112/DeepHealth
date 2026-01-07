
from losses import LogNormalBasisHazardLoss, ExponentialNLLLoss, _gauss_legendre_16
from model import DelphiFork, SapDelphi
from dataset import HealthDataset, health_collate_fn
import os
import argparse
import sys
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index

# Add current directory to path so we can import modules
sys.path.append(os.getcwd())


# =============================================================================
# Helper Functions for Evaluation
# =============================================================================


def load_labels(labels_file="labels.csv"):
    """
    Load disease labels mapping ID -> Name.
    ID starts at 2 (0=pad, 1=DOA).
    """
    labels_map = {}
    if not os.path.exists(labels_file):
        return labels_map
    with open(labels_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            parts = line.strip()
            if parts:
                labels_map[idx + 2] = parts
    return labels_map


def calculate_conditional_risk(theta, t_start, t_end, loss_type, loss_fn=None):
    """
    Calculate P(t_start < T <= t_end | T > t_start) using CDF strictly.
    Formula: (F(t_end) - F(t_start)) / (1 - F(t_start))
    """
    device = theta.device

    if loss_type == "exponential":
        logits = theta
        if logits.dim() == 3:
            logits = logits.squeeze(-1)

        lambdas = F.softplus(logits)  # (B, K)

        # Exponential CDF: F(t) = 1 - exp(-lambda * t)
        # We need to broadcast t_start/t_end to (B, K) if they are scalars
        # But actually we can just work with the exponents.

        # F(te) = 1 - exp(-lambda * te)
        # F(ts) = 1 - exp(-lambda * ts)
        # Numerator = (1 - exp(-lambda * te)) - (1 - exp(-lambda * ts))
        #           = exp(-lambda * ts) - exp(-lambda * te)
        # Denom = 1 - (1 - exp(-lambda * ts)) = exp(-lambda * ts)
        # Ratio = 1 - exp(-lambda * te) / exp(-lambda * ts) = 1 - exp(-lambda*(te-ts))

        # The user requested using F explicitly.
        dt = t_end - t_start
        if isinstance(dt, torch.Tensor) and dt.ndim == 1:
            dt = dt.unsqueeze(-1)  # (B, 1)

        risk = 1.0 - torch.exp(-lambdas * dt)
        return risk

    elif loss_type == "lognormal":
        if loss_fn is None:
            raise ValueError(
                "loss_fn must be provided for lognormal conditional risk")

        coeffs = theta  # (B, K, n_basis)

        def compute_F(t_vals):
            # t_vals: (B,) or scalar broadcasted
            if isinstance(t_vals, float):
                t_vals = torch.full((coeffs.shape[0],), t_vals, device=device)

            t = torch.clamp(t_vals, min=loss_fn.eps)

            # Integration for H(t)
            x_nodes, w = _gauss_legendre_16(
                device=device, dtype=coeffs.dtype)  # (16,)

            u = (t.unsqueeze(1) / 2.0) * (x_nodes.unsqueeze(0) + 1.0)
            weights = (t.unsqueeze(1) / 2.0) * w.unsqueeze(0)  # (B, 16)

            u_flat = u.reshape(-1)
            K_u_flat = loss_fn._compute_kernel(u_flat)
            K_u = K_u_flat.view(u.shape[0], u.shape[1], -1)

            log_hazards_u = torch.einsum("mkb,mqb->mkq", coeffs, K_u)
            log_hazards_u = torch.clamp(log_hazards_u, max=20.0)
            hazards_u = torch.exp(log_hazards_u)

            H = torch.sum(hazards_u * weights.unsqueeze(1), dim=2)  # (B, K)

            # F(t) = 1 - exp(-H(t))
            F = 1.0 - torch.exp(-H)
            return F

        F_start = compute_F(t_start)
        F_end = compute_F(t_end)

        # Formula: (F(te) - F(ts)) / (1 - F(ts))
        # Clamp F_start < 1.0 slightly to avoid division by zero
        F_start = torch.clamp(F_start, max=1.0 - 1e-7)

        # Numerator
        num = F_end - F_start
        num = torch.clamp(num, min=0.0)  # Ensure non-negative probability

        denom = 1.0 - F_start

        risk = num / denom
        return risk

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


# =============================================================================
# Experiment Runners
# =============================================================================

def run_stratified_evaluation(dataset, model, loss_fn, args, disease_map):
    print("\n>>> Starting Experiment 1: Age-Stratified Evaluation")

    age_buckets = [40, 45, 50, 55, 60, 65, 70, 75, 80]
    results = []

    # Identify Death Code from map
    # Reverse map: Name -> ID
    death_id = None
    for k, v in disease_map.items():
        if v == "Death":
            death_id = k
            break

    print(f"Death ID identified as: {death_id}")

    # Handle Subset or original Dataset
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        def get_real_idx(i): return dataset.indices[i]
    else:
        base_dataset = dataset
        def get_real_idx(i): return i

    # Pre-select patients for efficiency
    patient_meta = []
    for i in range(len(dataset)):
        real_idx = get_real_idx(i)
        pid = base_dataset.patient_ids[real_idx]
        events = sorted(base_dataset.patient_events[pid], key=lambda x: x[0])
        ts = [x[0]/365.25 for x in events]
        es = [x[1] for x in events]

        death_time = None
        if death_id is not None:
            # Check if death event exists
            for t_val, e_code in zip(ts, es):
                if e_code == death_id:
                    death_time = t_val
                    break

        patient_meta.append({
            'idx': i,
            'max_time': ts[-1] if ts else 0,
            'death_time': death_time,
            'events': list(zip(ts, es))
        })

    # Helper for ICD Chapters
    def get_chapter(code_str):
        if not code_str:
            return "Unknown"
        letter = code_str[0].upper()
        if letter in ['A', 'B']:
            return "I: Infections (A-B)"
        if letter in ['C']:
            return "II: Neoplasms (C)"
        if letter in ['D']:
            return "II/III: Neoplasms/Blood (D)"
        if letter in ['E']:
            return "IV: Endocrine (E)"
        if letter in ['F']:
            return "V: Mental (F)"
        if letter in ['G']:
            return "VI: Nervous (G)"
        if letter in ['H']:
            return "VII/VIII: Eye/Ear (H)"
        if letter in ['I']:
            return "IX: Circulatory (I)"
        if letter in ['J']:
            return "X: Respiratory (J)"
        if letter in ['K']:
            return "XI: Digestive (K)"
        if letter in ['L']:
            return "XII: Skin (L)"
        if letter in ['M']:
            return "XIII: Musculoskeletal (M)"
        if letter in ['N']:
            return "XIV: Genitourinary (N)"
        if letter in ['O']:
            return "XV: Pregnancy (O)"
        if letter in ['P']:
            return "XVI: Perinatal (P)"
        if letter in ['Q']:
            return "XVII: Congenial (Q)"
        if letter in ['R']:
            return "XVIII: Symptoms (R)"
        if letter in ['S', 'T']:
            return "XIX: Injury (S-T)"
        if letter in ['V', 'W', 'X', 'Y']:
            return "XX: External (V-Y)"
        if letter in ['Z']:
            return "XXI: Factors (Z)"
        return "Other"

    for age in age_buckets:
        t_start = float(age)
        t_target_start = t_start + 0.5
        t_target_end = t_start + 5.0

        eligible_indices = []
        for pm in patient_meta:
            # Strict Filtering:
            # 1. Observed until end of window (max_time >= t_target_end)
            # 2. OR Died strictly within the window (t_target_start < death_time <= t_target_end)
            # Note: If died < t_target_start, they are not at risk at t_start?
            # Actually, standard survival at age A requires being alive at A.
            # So if death_time <= t_start, they are already dead -> Exclude.
            # If death_time > t_target_end, they are covered by max_time >= t_target_end check.

            # First check: Alive at t_start?
            if pm['death_time'] is not None and pm['death_time'] <= t_start:
                continue

            # Then check followup sufficiency
            is_followed_up = pm['max_time'] >= t_target_end
            is_dead_in_window = (pm['death_time'] is not None and
                                 pm['death_time'] > t_start and
                                 pm['death_time'] <= t_target_end)

            # Wait, if death_time is in (t_start, t_target_start], does that count?
            # We are predicting "disease in (A+0.5, A+5]".
            # If they die at A+0.2, they can't get disease in window -> Outcome 0 (valid).
            # So strictly: death > t_start is sufficient to close the "Outcome 0" case?
            # Yes, if they die, the risk of disease is 0 (competing risk).
            # So as long as death_time > t_start, they are valid candidates with outcome 0.
            # But we need to distinguish "Dropout" (censored) from "Death" (valid 0).
            # Code below checks "is_dead_in_window" relative to prediction horizon end.
            # If they die at t_start + 0.2 (before window start), they are a valid 0 for the window [A+0.5, A+5].
            # So simple logic: EITHER followed up >= t_target_end OR Died after t_start.

            # Refined:
            # 1. Must be alive at t_start.
            # 2. Must be fully observed until t_target_end OR have a terminating event (Death) after t_start.
            if is_followed_up or (pm['death_time'] is not None and pm['death_time'] > t_start):
                eligible_indices.append(pm['idx'])

        if not eligible_indices:
            print(f"Age {age}: No eligible patients.")
            continue

        print(f"Age {age}: {len(eligible_indices)} patients.")

        loader = DataLoader(
            Subset(dataset, eligible_indices),
            batch_size=args.batch_size,
            collate_fn=health_collate_fn,
            shuffle=False
        )

        all_risks = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                event_batch, time_batch, cont_batch, cate_batch, sex_batch = batch

                limit_days = t_start * 365.25
                new_event_seqs = []
                new_time_seqs = []
                batch_targets = []

                B_curr = event_batch.shape[0]

                kept_indices = []
                for k in range(B_curr):
                    valid_len = (event_batch[k] != 0).sum().item()
                    e_seq = event_batch[k, :valid_len]
                    t_seq = time_batch[k, :valid_len]

                    mask = t_seq <= limit_days
                    t_trunc = t_seq[mask]
                    e_trunc = e_seq[mask]

                    if len(t_trunc) == 0:
                        continue

                    kept_indices.append(k)

                    # Removed artificial appending of DOA token per user request.
                    # t_trunc = torch.cat([t_trunc, torch.tensor([limit_days],  dtype=t_trunc.dtype)])
                    # e_trunc = torch.cat([e_trunc, torch.tensor([1], dtype=e_trunc.dtype)])

                    new_event_seqs.append(e_trunc)
                    new_time_seqs.append(t_trunc)

                    target_start_days = t_target_start * 365.25
                    target_end_days = t_target_end * 365.25

                    in_window_mask = (t_seq > target_start_days) & (
                        t_seq <= target_end_days)
                    target_events = e_seq[in_window_mask].tolist()

                    tgt_vec = np.zeros(base_dataset.n_disease, dtype=float)
                    for eid in target_events:
                        if eid >= 2:
                            idx_in_vec = eid - 2
                            if idx_in_vec < base_dataset.n_disease:
                                tgt_vec[idx_in_vec] = 1.0
                    batch_targets.append(tgt_vec)

                if len(new_event_seqs) == 0:
                    continue

                from torch.nn.utils.rnn import pad_sequence
                event_in = pad_sequence(
                    new_event_seqs, batch_first=True, padding_value=0).to(args.device)
                time_in = pad_sequence(
                    new_time_seqs, batch_first=True, padding_value=36525.0).to(args.device)

                kept_indices_tensor = torch.tensor(
                    kept_indices, device=cont_batch.device)
                cont_in = cont_batch[kept_indices_tensor].to(args.device)
                cate_in = cate_batch[kept_indices_tensor].to(args.device)
                sex_in = sex_batch[kept_indices_tensor].to(args.device)

                lengths = torch.tensor(
                    [len(x) for x in new_event_seqs], device=args.device)
                b_prev = torch.arange(len(new_event_seqs), device=args.device)
                t_prev = lengths - 1

                theta = model(event_in, time_in, sex_in, cont_in,
                              cate_in, b_prev=b_prev, t_prev=t_prev)

                n_dim = model.n_dim
                theta = theta.view(len(new_event_seqs),
                                   base_dataset.n_disease, n_dim)

                # Correct logic for Time Anchor Mismatch:
                # The model predicts risk starting from t_last (last event time).
                # We want risk in window [Anchor+0.5, Anchor+5.0].
                # So relative to t_last, the window is [Anchor-t_last + 0.5, Anchor-t_last + 5.0].

                # Get last event time for each patient in batch (in days)
                last_times_days = torch.tensor(
                    [t[-1] for t in new_time_seqs], device=args.device)
                anchor_days = limit_days
                gap_days = anchor_days - last_times_days
                gap_years = gap_days / 365.25

                # Compute risk window relative to last event
                risk_t_start = gap_years + 0.5
                risk_t_end = gap_years + 5.0

                risks = calculate_conditional_risk(
                    theta,
                    t_start=risk_t_start,
                    t_end=risk_t_end,
                    loss_type=loss_type,
                    loss_fn=loss_fn
                )

                all_risks.append(risks.cpu().numpy())
                all_targets.append(np.array(batch_targets))

        all_risks = np.concatenate(all_risks, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate Metrics per Disease
        auc_scores = {}  # d_idx -> auc

        print(f"Evaluating Age {age} metrics...")
        for d_idx in range(base_dataset.n_disease):
            y_true = all_targets[:, d_idx]
            y_score = all_risks[:, d_idx]

            if np.sum(y_true) < 5:
                continue

            try:
                auc = roc_auc_score(y_true, y_score)
                auc_scores[d_idx] = auc
            except:
                pass

        # Aggregate Results
        valid_aucs = list(auc_scores.values())
        mean_auc = np.mean(valid_aucs) if valid_aucs else 0.0

        print(f"Age {age} Mean AUC: {mean_auc:.4f}")

        res_row = {'Age': age, 'Mean_AUC': mean_auc,
                   'N_Patients': len(eligible_indices)}

        # Chapter Aggregation
        chapter_aucs = defaultdict(list)
        for d_idx, auc in auc_scores.items():
            disease_id = d_idx + 2
            name = disease_map.get(disease_id, "Unknown")
            chap = get_chapter(name)
            chapter_aucs[chap].append(auc)

        for chap, aucs in chapter_aucs.items():
            res_row[f"Chapter_{chap}_AUC"] = np.mean(aucs)

        results.append(res_row)

    df_res = pd.DataFrame(results)
    df_res.to_csv("results_exp1_stratified.csv", index=False)
    print("Saved results_exp1_stratified.csv")


def run_landmark_analysis(dataset, model, loss_fn, args):
    print("\n>>> Starting Experiment 2: Age-60 Landmark Analysis")

    t_landmark = 60.0
    limit_days = t_landmark * 365.25

    # Handle Subset or original Dataset
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        def get_real_idx(i): return dataset.indices[i]
    else:
        base_dataset = dataset
        def get_real_idx(i): return i

    eligible_indices = []

    for i in range(len(dataset)):
        real_idx = get_real_idx(i)
        pid = base_dataset.patient_ids[real_idx]
        events = base_dataset.patient_events[pid]
        has_future = any(x[0] > limit_days for x in events)
        if has_future:
            eligible_indices.append(i)

    print(f"Found {len(eligible_indices)} patients for landmark analysis.")
    if len(eligible_indices) == 0:
        return

    loader = DataLoader(
        Subset(dataset, eligible_indices),
        batch_size=args.batch_size,
        collate_fn=health_collate_fn,
        shuffle=False
    )

    horizons = [5, 10, 20]
    calibration_data = {h: [] for h in horizons}
    ranking_data = {h: [] for h in horizons}

    all_pred_risks = {h: [] for h in horizons}
    all_true_outcomes = {h: [] for h in horizons}

    with torch.no_grad():
        for batch in loader:
            event_batch, time_batch, cont_batch, cate_batch, sex_batch = batch
            new_event_seqs = []
            new_time_seqs = []
            batch_targets = {h: [] for h in horizons}

            B_curr = event_batch.shape[0]
            kept_indices = []
            for k in range(B_curr):
                valid_len = (event_batch[k] != 0).sum().item()
                e_seq = event_batch[k, :valid_len]
                t_seq = time_batch[k, :valid_len]

                mask = t_seq <= limit_days
                t_trunc = t_seq[mask]
                e_trunc = e_seq[mask]

                if len(t_trunc) == 0:
                    continue
                kept_indices.append(k)

                # Removed artificial appending of DOA per user request regarding Experiment 1.
                # Assuming Experiment 2 should follow same rigorous truncation.
                # t_trunc = torch.cat([t_trunc, torch.tensor([limit_days],  dtype=t_trunc.dtype)])
                # e_trunc = torch.cat([e_trunc, torch.tensor([1], dtype=e_trunc.dtype)])

                new_event_seqs.append(e_trunc)
                new_time_seqs.append(t_trunc)

                for h in horizons:
                    h_days = (t_landmark + h) * 365.25
                    in_window = (t_seq > limit_days) & (t_seq <= h_days)
                    tgt_codes = e_seq[in_window].tolist()

                    vec = np.zeros(base_dataset.n_disease, dtype=float)
                    for c in tgt_codes:
                        if c >= 2 and c-2 < base_dataset.n_disease:
                            vec[c-2] = 1.0
                    batch_targets[h].append(vec)

            if len(new_event_seqs) == 0:
                continue

            from torch.nn.utils.rnn import pad_sequence
            event_in = pad_sequence(
                new_event_seqs, batch_first=True, padding_value=0).to(args.device)
            time_in = pad_sequence(
                new_time_seqs, batch_first=True, padding_value=36525.0).to(args.device)

            kept_indices_tensor = torch.tensor(
                kept_indices, device=cont_batch.device)
            cont_in = cont_batch[kept_indices_tensor].to(args.device)
            cate_in = cate_batch[kept_indices_tensor].to(args.device)
            sex_in = sex_batch[kept_indices_tensor].to(args.device)

            lengths = torch.tensor([len(x)
                                   for x in new_event_seqs], device=args.device)
            b_prev = torch.arange(len(new_event_seqs), device=args.device)
            t_prev = lengths - 1

            theta = model(event_in, time_in, sex_in, cont_in,
                          cate_in, b_prev=b_prev, t_prev=t_prev)
            theta = theta.view(len(new_event_seqs),
                               base_dataset.n_disease, model.n_dim)

            # Correct logic for Time Anchor Mismatch:
            # Anchor is fixed at limit_days (Age 60).
            # Gap = 60 - Last_Event_Time.
            # Window starts at Gap + 0, ends at Gap + Horizon.

            last_times_days = torch.tensor(
                [t[-1] for t in new_time_seqs], device=args.device)
            gap_days = limit_days - last_times_days
            gap_years = gap_days / 365.25

            for h in horizons:
                risk_t_start = gap_years + 0.0
                risk_t_end = gap_years + float(h)

                risks = calculate_conditional_risk(
                    theta,
                    t_start=risk_t_start,
                    t_end=risk_t_end,
                    loss_type=loss_type,
                    loss_fn=loss_fn)

                all_pred_risks[h].append(risks.cpu().numpy())
                all_true_outcomes[h].append(np.array(batch_targets[h]))

    for h in horizons:
        preds = np.concatenate(all_pred_risks[h], axis=0)
        trues = np.concatenate(all_true_outcomes[h], axis=0)

        total_expected = np.sum(preds)
        total_observed = np.sum(trues)
        eo_ratio = total_expected / (total_observed + 1e-8)

        print(f"Horizon {h} Years: E/O Ratio = {eo_ratio:.4f}")

        calibration_data[h] = {
            'mean_pred': np.mean(preds),
            'mean_obs': np.mean(trues),
            'eo_ratio': eo_ratio
        }

        ks = [10, 20]
        recalls = {k: [] for k in ks}

        for i in range(preds.shape[0]):
            p_row = preds[i]
            t_row = trues[i]

            relevant_indices = np.where(t_row > 0)[0]
            if len(relevant_indices) == 0:
                continue

            sorted_indices = np.argsort(p_row)[::-1]
            for k in ks:
                top_k = sorted_indices[:k]
                hits = np.isin(top_k, relevant_indices).sum()
                recall = hits / len(relevant_indices)
                recalls[k].append(recall)

        avg_recalls = {k: np.mean(v) if v else 0.0 for k, v in recalls.items()}
        print(
            f"Horizon {h} Recall@10: {avg_recalls[10]:.4f}, Recall@20: {avg_recalls[20]:.4f}")

        ranking_row = {'Horizon': h, 'EO_Ratio': eo_ratio,
                       'Recall_10': avg_recalls[10], 'Recall_20': avg_recalls[20]}
        ranking_data[h] = ranking_row

    df_cal = pd.DataFrame([calibration_data[h] for h in horizons])
    df_cal['Horizon'] = horizons
    df_cal.to_csv("results_exp2_calibration.csv", index=False)

    df_rank = pd.DataFrame([ranking_data[h] for h in horizons])
    df_rank.to_csv("results_exp2_ranking.csv", index=False)
    print("Saved results_exp2_ranking.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--loss_type", type=str,
                        choices=['lognormal', 'exponential'], default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    config_path = os.path.join(args.run_dir, "train_config.json")
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)
    print(f"Loaded config from {args.run_dir}")

    loss_type = args.loss_type
    if loss_type is None:
        loss_type = cfg_dict.get("loss_type")
        if loss_type is None:
            raise ValueError(
                "loss_type not found in config and not provided via argument.")
        print(f"Inferred loss_type: {loss_type}")

    data_prefix = cfg_dict.get("data_prefix", "ukb")
    if cfg_dict.get("full_cov", False):
        cov_list = None
    else:
        cov_list = ["bmi", "smoking", "alcohol"]

    print("Loading dataset...")
    # Check simple file existence for quick failure
    if not os.path.exists(f"{data_prefix}_basic_info.csv"):
        print(
            f"Error: {data_prefix}_basic_info.csv not found. Please run prepare_data.py.")
        sys.exit(1)

    dataset = HealthDataset(data_prefix=data_prefix, covariate_list=cov_list)
    print(f"Full Dataset size: {len(dataset)}")

    # Split dataset based on config
    n_total = len(dataset)
    train_ratio = cfg_dict.get("train_ratio", 0.7)
    val_ratio = cfg_dict.get("val_ratio", 0.15)
    test_ratio = 1.0 - train_ratio - val_ratio
    random_seed = cfg_dict.get("random_seed", 42)

    train_len = int(n_total * train_ratio)
    val_len = int(n_total * val_ratio)
    test_len = n_total - train_len - val_len

    _, _, test_dataset = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(random_seed)
    )
    print(f"Using test split with {len(test_dataset)} samples.")

    # Use the test subset for evaluation
    eval_dataset = test_dataset

    n_dim = 1
    loss_fn = None

    if loss_type == "exponential":
        n_dim = 1
        loss_fn = ExponentialNLLLoss().to(args.device)
    elif loss_type == "lognormal":
        bin_edges = (
            0.010951, 0.090349, 0.238193, 0.443532, 0.722793, 1.070500,
            1.612594, 2.409309, 3.841205, 7.000684, 30.997947
        )
        centers = list(bin_edges)
        n_dim = len(centers)
        loss_fn = LogNormalBasisHazardLoss(centers=centers).to(args.device)

    model_type = cfg_dict.get("model_type", "delphifork")
    n_embd = cfg_dict.get("n_embd", 120)

    if model_type == "delphifork":
        model = DelphiFork(
            n_disease=dataset.n_disease,
            n_tech_tokens=2,
            n_cont=dataset.n_cont,
            n_cate=dataset.n_cate,
            cate_dims=dataset.cate_dims,
            n_embd=n_embd,
            n_layer=cfg_dict.get("n_layer", 12),
            n_head=cfg_dict.get("n_head", 12),
            pdrop=0.0,
            age_encoder_type=cfg_dict.get("age_encoder", "sinusoidal"),
            n_dim=n_dim
        )
    elif model_type == "sapdelphi":
        model = SapDelphi(
            n_disease=dataset.n_disease,
            n_tech_tokens=2,
            n_cont=dataset.n_cont,
            n_cate=dataset.n_cate,
            cate_dims=dataset.cate_dims,
            n_embd=n_embd,
            n_layer=cfg_dict.get("n_layer", 12),
            n_head=cfg_dict.get("n_head", 12),
            pdrop=0.0,
            age_encoder_type=cfg_dict.get("age_encoder", "sinusoidal"),
            n_dim=n_dim,
            pretrained_weights_path=cfg_dict.get("pretrained_weights_path"),
        )

    model.to(args.device)

    best_path = os.path.join(args.run_dir, "best_model.pt")
    if not os.path.exists(best_path):
        best_path = os.path.join(args.run_dir, "last_model.pt")

    print(f"Loading weights from {best_path}...")
    ckpt = torch.load(best_path, map_location=args.device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        if "criterion_state_dict" in ckpt and loss_fn is not None:
            print("Loading criterion state (sigma)...")
            loss_fn.load_state_dict(ckpt["criterion_state_dict"])
        else:
            print("Warning: No criterion_state_dict found.")
    else:
        model.load_state_dict(ckpt)

    model.eval()

    disease_map = load_labels()

    run_stratified_evaluation(eval_dataset, model, loss_fn, args, disease_map)
    run_landmark_analysis(eval_dataset, model, loss_fn, args)

    print("Evaluation Complete.")
