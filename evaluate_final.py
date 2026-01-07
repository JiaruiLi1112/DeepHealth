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
# Utility Functions
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
                # ID starts at 2 (0=PAD, 1=DOA)
                labels_map[idx + 2] = parts
    return labels_map


def get_chapter(code_str):
    """Map disease name/code to ICD-10 Chapter."""
    if not code_str:
        return "Unknown"
    # Simple heuristic mapping based on first letter
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
    # Special handling for "Death" if it's explicitly named
    if "Death" in code_str:
        return "Death"
    return mapping.get(letter, "Other")


def calculate_conditional_risk(theta, t_start, t_end, loss_type, loss_fn=None):
    """
    Calculate P(t_start < T <= t_end | T > t_start).
    This assumes t_start and t_end are relative to the last observed event.
    Formula: (F(t_end) - F(t_start)) / (1 - F(t_start))
    """
    device = theta.device

    if loss_type == "exponential":
        # theta is logits -> lambda = softplus(logits)
        logits = theta
        if logits.dim() == 3:
            logits = logits.squeeze(-1)
        lambdas = F.softplus(logits)  # (B, K)

        dt = t_end - t_start
        if isinstance(dt, torch.Tensor) and dt.ndim == 1:
            dt = dt.unsqueeze(-1)  # (B, 1)

        # For exponential, hazard is constant, so conditional prob depends only on dt
        risk = 1.0 - torch.exp(-lambdas * dt)
        return risk

    elif loss_type == "lognormal":
        if loss_fn is None:
            raise ValueError("loss_fn required for lognormal risk")

        coeffs = theta  # (B, K, n_basis)

        def compute_F(t_vals):
            # Broadcast t_vals to (B,)
            if isinstance(t_vals, float):
                t_vals = torch.full((coeffs.shape[0],), t_vals, device=device)

            # Clamp for numerical stability
            t = torch.clamp(t_vals, min=1e-5)

            # Gauss-Legendre Integration for Cumulative Hazard H(t)
            x_nodes, w = _gauss_legendre_16(device=device, dtype=coeffs.dtype)

            # Map [-1, 1] to [0, t]
            u = (t.unsqueeze(1) / 2.0) * (x_nodes.unsqueeze(0) + 1.0)  # (B, 16)
            weights = (t.unsqueeze(1) / 2.0) * w.unsqueeze(0)         # (B, 16)

            u_flat = u.reshape(-1)
            K_u_flat = loss_fn._compute_kernel(u_flat)
            K_u = K_u_flat.view(u.shape[0], u.shape[1], -1)  # (B, 16, n_basis)

            # log_lambda(u)
            log_hazards_u = torch.einsum("mkb,mqb->mkq", coeffs, K_u)
            log_hazards_u = torch.clamp(log_hazards_u, max=20.0)
            hazards_u = torch.exp(log_hazards_u)  # (B, K, 16)

            # Integrate -> H(t)
            H = torch.sum(hazards_u * weights.unsqueeze(1), dim=2)  # (B, K)

            # F(t) = 1 - exp(-H(t))
            return 1.0 - torch.exp(-H)

        F_start = compute_F(t_start)
        F_end = compute_F(t_end)

        # Conditional Probability Formula
        # Clamp F_start to avoid division by zero
        F_start = torch.clamp(F_start, max=1.0 - 1e-7)

        num = F_end - F_start
        num = torch.clamp(num, min=0.0)
        denom = 1.0 - F_start

        return num / denom

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

# =============================================================================
# Experiment 1: Age-Stratified Evaluation
# =============================================================================


def run_stratified_evaluation(dataset, model, loss_fn, args, disease_map):
    print("\n" + "="*50)
    print(">>> Experiment 1: Age-Stratified Evaluation")
    print("="*50)

    # 1. Setup Metadata for fast filtering
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        def get_real_idx(i): return dataset.indices[i]
    else:
        base_dataset = dataset
        def get_real_idx(i): return i

    print("Pre-scanning patient metadata...")
    patient_meta = []
    # Find Death ID
    death_id = None
    for k, v in disease_map.items():
        if "Death" in v or "death" in v:
            death_id = k
            break

    for i in range(len(dataset)):
        pid = base_dataset.patient_ids[get_real_idx(i)]
        events = base_dataset.patient_events[pid]  # list of (time, code)
        if not events:
            patient_meta.append(
                {'idx': i, 'max_time_yr': 0, 'death_time_yr': None})
            continue

        # Sort by time
        events = sorted(events, key=lambda x: x[0])
        max_time_yr = events[-1][0] / 365.25

        d_time = None
        if death_id is not None:
            for t, c in events:
                if c == death_id:
                    d_time = t / 365.25
                    break

        patient_meta.append({
            'idx': i,
            'max_time_yr': max_time_yr,
            'death_time_yr': d_time
        })

    age_buckets = [40, 45, 50, 55, 60, 65, 70, 75, 80]
    results = []

    for age in age_buckets:
        t_start = float(age)
        t_gap = 0.5
        t_horizon = 5.0
        t_end_window = t_start + t_horizon

        # 2. Strict Filtering (Censoring Handling)
        # Patient must be alive at start AND (followed up until end OR died within window)
        eligible_indices = []
        for pm in patient_meta:
            # Must be alive at start
            if pm['death_time_yr'] is not None and pm['death_time_yr'] <= t_start:
                continue

            # Check censoring
            # Condition: Max observation >= End of Window OR Death happened during window
            has_sufficient_followup = pm['max_time_yr'] >= t_end_window
            died_in_window = (pm['death_time_yr'] is not None) and (
                pm['death_time_yr'] <= t_end_window)

            if has_sufficient_followup or died_in_window:
                eligible_indices.append(pm['idx'])

        if len(eligible_indices) < 50:
            print(
                f"Age {age}: Skipped (insufficient samples: {len(eligible_indices)})")
            continue

        print(f"Age {age}: Evaluating on {len(eligible_indices)} patients...")

        loader = DataLoader(
            Subset(dataset, eligible_indices),
            batch_size=args.batch_size,
            collate_fn=health_collate_fn,
            shuffle=False
        )

        all_risks = []
        all_targets = []
        all_prevalence = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Age {age} Inference", leave=False):
                event_batch, time_batch, cont_batch, cate_batch, sex_batch = batch

                # Truncate at t_start
                limit_days = t_start * 365.25
                target_start_days = (t_start + t_gap) * 365.25
                target_end_days = (t_start + t_horizon) * 365.25

                new_event_seqs = []
                new_time_seqs = []
                batch_targets = []
                batch_prevalence = []
                kept_indices = []

                B = event_batch.shape[0]
                for k in range(B):
                    # Get valid sequence
                    valid_len = (event_batch[k] != 0).sum().item()
                    e_seq = event_batch[k, :valid_len]
                    t_seq = time_batch[k, :valid_len]

                    # 1. Truncate History (Input)
                    mask_input = t_seq <= limit_days
                    t_trunc = t_seq[mask_input]
                    e_trunc = e_seq[mask_input]

                    if len(t_trunc) == 0:
                        continue

                    # 2. Define Target (Future Events)
                    # Strictly inside (Age + 0.5, Age + 5.0]
                    mask_target = (t_seq > target_start_days) & (
                        t_seq <= target_end_days)
                    target_codes = e_seq[mask_target].tolist()

                    tgt_vec = np.zeros(base_dataset.n_disease, dtype=float)
                    for c in target_codes:
                        if c >= 2:
                            tgt_vec[c - 2] = 1.0

                    # 3. Compute Prevalence (History)
                    # Identify diseases present in input history
                    prev_vec = np.zeros(base_dataset.n_disease, dtype=bool)
                    for c in e_trunc:
                        if c >= 2:
                            prev_vec[c - 2] = True

                    new_event_seqs.append(e_trunc)
                    new_time_seqs.append(t_trunc)
                    batch_targets.append(tgt_vec)
                    batch_prevalence.append(prev_vec)
                    kept_indices.append(k)

                if not new_event_seqs:
                    continue

                # Prepare Model Inputs
                from torch.nn.utils.rnn import pad_sequence
                event_in = pad_sequence(
                    new_event_seqs, batch_first=True, padding_value=0).to(args.device)
                time_in = pad_sequence(
                    new_time_seqs, batch_first=True, padding_value=36525.0).to(args.device)

                kept_tensor = torch.tensor(
                    kept_indices, device=cont_batch.device)
                cont_in = cont_batch[kept_tensor].to(args.device)
                cate_in = cate_batch[kept_tensor].to(args.device)
                sex_in = sex_batch[kept_tensor].to(args.device)

                b_prev = torch.arange(len(new_event_seqs), device=args.device)
                t_prev = torch.tensor(
                    [len(x)-1 for x in new_event_seqs], device=args.device)

                # Forward Pass
                theta = model(event_in, time_in, sex_in, cont_in,
                              cate_in, b_prev=b_prev, t_prev=t_prev)
                theta = theta.view(len(new_event_seqs),
                                   base_dataset.n_disease, -1)

                # Time Logic (Relative to Last Event)
                last_times = torch.tensor(
                    [t[-1] for t in new_time_seqs], device=args.device)
                last_ages_years = last_times / 365.25

                # Gap = Anchor Age - Last Event Age
                gap_years = t_start - last_ages_years
                gap_years = torch.clamp(gap_years, min=0.0)  # Safety

                # Prediction Interval: [Gap + 0.5, Gap + 5.0]
                risk_t_start = gap_years + t_gap
                risk_t_end = gap_years + t_horizon

                risks = calculate_conditional_risk(
                    theta, risk_t_start, risk_t_end, args.loss_type, loss_fn
                )

                all_risks.append(risks.cpu().numpy())
                all_targets.append(np.array(batch_targets))
                all_prevalence.append(np.array(batch_prevalence))

        if not all_risks:
            continue

        all_risks = np.concatenate(all_risks, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_prevalence = np.concatenate(all_prevalence, axis=0)

        # Compute AUCs with Prevalence Masking
        auc_dict = {}
        for d_idx in range(base_dataset.n_disease):
            # Only evaluate on patients who DO NOT have this disease in history
            at_risk_mask = ~all_prevalence[:, d_idx]

            y_true = all_targets[at_risk_mask, d_idx]
            y_score = all_risks[at_risk_mask, d_idx]

            # Need at least one class 0 and one class 1
            if len(y_true) < 10 or np.sum(y_true) < 1 or np.sum(y_true) == len(y_true):
                continue

            try:
                auc_dict[d_idx] = roc_auc_score(y_true, y_score)
            except:
                pass

        # Aggregate Metrics
        valid_aucs = list(auc_dict.values())
        mean_auc = np.mean(valid_aucs) if valid_aucs else 0.0

        row = {
            "Age": age,
            "N_Patients": len(eligible_indices),
            "Mean_AUC": mean_auc
        }

        # Aggregation by Chapter
        chap_scores = defaultdict(list)
        for d_idx, auc_val in auc_dict.items():
            name = disease_map.get(d_idx+2, "Unknown")
            chap = get_chapter(name)
            chap_scores[chap].append(auc_val)

        for chap, scores in chap_scores.items():
            row[f"AUC_{chap}"] = np.mean(scores)

        results.append(row)
        print(f"  -> Mean AUC: {mean_auc:.4f}")

    # Save
    df = pd.DataFrame(results)
    out_path = os.path.join(args.run_dir, "results_exp1_stratified.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved stratified results to {out_path}")


# =============================================================================
# Experiment 2: Age-60 Landmark Analysis
# =============================================================================

def run_landmark_analysis(dataset, model, loss_fn, args):
    print("\n" + "="*50)
    print(">>> Experiment 2: Age-60 Landmark Analysis")
    print("="*50)

    t_landmark = 60.0
    limit_days = t_landmark * 365.25
    gap_years = 1.0  # 1-Year Blanking
    min_future_days = (t_landmark + gap_years) * 365.25

    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        def get_real_idx(i): return dataset.indices[i]
    else:
        base_dataset = dataset
        def get_real_idx(i): return i

    # Filter: Must have data beyond 61 years
    eligible_indices = []
    for i in range(len(dataset)):
        pid = base_dataset.patient_ids[get_real_idx(i)]
        events = base_dataset.patient_events[pid]
        # Check if max time > 61 years
        if events and events[-1][0] > min_future_days:
            eligible_indices.append(i)

    print(f"Found {len(eligible_indices)} patients eligible (Follow-up > 61y).")
    if not eligible_indices:
        return

    loader = DataLoader(
        Subset(dataset, eligible_indices),
        batch_size=args.batch_size,
        collate_fn=health_collate_fn,
        shuffle=False
    )

    horizons = [5, 10, 20]

    # Store predictions and targets for all patients
    # Use lists to accumulate batches
    collated_preds = {h: [] for h in horizons}
    collated_targets = {h: [] for h in horizons}
    collated_prevalence = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Landmark Inference"):
            event_batch, time_batch, cont_batch, cate_batch, sex_batch = batch

            new_event_seqs = []
            new_time_seqs = []
            kept_indices = []

            batch_targets = {h: [] for h in horizons}
            batch_prev = []

            B = event_batch.shape[0]
            for k in range(B):
                valid_len = (event_batch[k] != 0).sum().item()
                e_seq = event_batch[k, :valid_len]
                t_seq = time_batch[k, :valid_len]

                # 1. Input: Truncate at 60.0
                mask_in = t_seq <= limit_days
                e_trunc = e_seq[mask_in]
                t_trunc = t_seq[mask_in]

                if len(t_trunc) == 0:
                    continue

                # 2. Prevalence Mask (History up to 60)
                prev_vec = np.zeros(base_dataset.n_disease, dtype=bool)
                for c in e_trunc:
                    if c >= 2:
                        prev_vec[c - 2] = True

                # 3. Targets for each horizon
                # Window: (61.0, 60.0 + H]
                start_days = min_future_days

                for h in horizons:
                    end_days = (t_landmark + h) * 365.25
                    mask_tgt = (t_seq > start_days) & (t_seq <= end_days)
                    tgt_codes = e_seq[mask_tgt].tolist()

                    tgt_vec = np.zeros(base_dataset.n_disease, dtype=float)
                    for c in tgt_codes:
                        if c >= 2:
                            tgt_vec[c - 2] = 1.0
                    batch_targets[h].append(tgt_vec)

                new_event_seqs.append(e_trunc)
                new_time_seqs.append(t_trunc)
                batch_prev.append(prev_vec)
                kept_indices.append(k)

            if not new_event_seqs:
                continue

            # Model Forward
            from torch.nn.utils.rnn import pad_sequence
            event_in = pad_sequence(
                new_event_seqs, batch_first=True).to(args.device)
            time_in = pad_sequence(
                new_time_seqs, batch_first=True, padding_value=36525.0).to(args.device)

            kept_t = torch.tensor(kept_indices, device=cont_batch.device)
            cont_in = cont_batch[kept_t].to(args.device)
            cate_in = cate_batch[kept_t].to(args.device)
            sex_in = sex_batch[kept_t].to(args.device)

            b_prev = torch.arange(len(new_event_seqs), device=args.device)
            t_prev = torch.tensor(
                [len(x)-1 for x in new_event_seqs], device=args.device)

            theta = model(event_in, time_in, sex_in, cont_in,
                          cate_in, b_prev=b_prev, t_prev=t_prev)
            theta = theta.view(len(new_event_seqs), base_dataset.n_disease, -1)

            # Time Logic:
            # We want to predict event in absolute window [61, 60+H]
            # Input last event time: t_last
            # Gap relative to t_last: (60 - t_last)
            # Prediction Start relative to t_last: (60 - t_last) + 1.0
            # Prediction End relative to t_last: (60 - t_last) + H

            last_times = torch.tensor(
                [t[-1] for t in new_time_seqs], device=args.device)
            last_ages = last_times / 365.25
            gap_to_60 = t_landmark - last_ages
            gap_to_60 = torch.clamp(gap_to_60, min=0.0)

            collated_prevalence.append(np.array(batch_prev))

            for h in horizons:
                t_rel_start = gap_to_60 + gap_years
                t_rel_end = gap_to_60 + float(h)

                risks = calculate_conditional_risk(
                    theta, t_rel_start, t_rel_end, args.loss_type, loss_fn
                )

                collated_preds[h].append(risks.cpu().numpy())
                collated_targets[h].append(np.array(batch_targets[h]))

    # Compute Metrics
    final_prevalence = np.concatenate(collated_prevalence, axis=0)  # (N, K)

    cal_results = []
    rank_results = []

    for h in horizons:
        if not collated_preds[h]:
            continue

        preds = np.concatenate(collated_preds[h], axis=0)  # (N, K)
        trues = np.concatenate(collated_targets[h], axis=0)  # (N, K)

        # 1. Global Calibration (E/O Ratio)
        # Sum expected risk vs Sum observed events
        # IMPORTANT: Only count people AT RISK (not prevalent)
        # Or typically E/O includes everyone?
        # Standard E/O usually includes everyone, but if we mask predictions for prevalent cases,
        # we should also mask observations (which are 0 anyway).

        # Let's apply prevalence masking to Predictions for E/O as well to be consistent
        # Mask: if prevalent, we set pred=0 (or ignore).
        # Since 'trues' is 0 for prevalent cases, this is safe.

        preds_masked = preds.copy()
        preds_masked[final_prevalence] = 0.0

        E = np.sum(preds_masked)
        O = np.sum(trues)
        eo = E / (O + 1e-8)

        print(f"Horizon {h}: E/O = {eo:.4f}")
        cal_results.append({'Horizon': h, 'Expected': E,
                           'Observed': O, 'EO_Ratio': eo})

        # 2. Ranking Metrics (Recall@K, Precision@K)
        # For each patient, rank diseases by risk.
        # EXCLUDE prevalent diseases from candidates (set risk to -1)

        preds_for_ranking = preds.copy()
        preds_for_ranking[final_prevalence] = -1.0

        recalls_10, precisions_10 = [], []
        recalls_20, precisions_20 = [], []

        for i in range(preds.shape[0]):
            true_indices = np.where(trues[i] > 0)[0]
            if len(true_indices) == 0:
                continue  # Patient had no events in window

            # Sort desc
            sorted_indices = np.argsort(preds_for_ranking[i])[::-1]

            # Top 10
            top10 = sorted_indices[:10]
            hits10 = np.isin(top10, true_indices).sum()
            recalls_10.append(hits10 / len(true_indices))
            precisions_10.append(hits10 / 10.0)

            # Top 20
            top20 = sorted_indices[:20]
            hits20 = np.isin(top20, true_indices).sum()
            recalls_20.append(hits20 / len(true_indices))
            precisions_20.append(hits20 / 20.0)

        rank_results.append({
            'Horizon': h,
            'Recall_10': np.mean(recalls_10), 'Precision_10': np.mean(precisions_10),
            'Recall_20': np.mean(recalls_20), 'Precision_20': np.mean(precisions_20)
        })

    # Save
    pd.DataFrame(cal_results).to_csv(os.path.join(
        args.run_dir, "results_exp2_calibration.csv"), index=False)
    pd.DataFrame(rank_results).to_csv(os.path.join(
        args.run_dir, "results_exp2_ranking.csv"), index=False)
    print("Saved landmark results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--loss_type", type=str, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # Load Config
    cfg_path = os.path.join(args.run_dir, "train_config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    args.loss_type = args.loss_type or cfg.get("loss_type", "lognormal")
    print(f"Run: {args.run_dir}, Loss: {args.loss_type}, Device: {args.device}")

    # Load Model
    n_dim = 1
    loss_fn = None
    if args.loss_type == "exponential":
        loss_fn = ExponentialNLLLoss()
    elif args.loss_type == "lognormal":
        # Safe Load Bin Edges
        bin_edges = cfg.get("bin_edges")
        if not bin_edges:
            print("WARNING: bin_edges not in config. Using DEFAULT.")
            bin_edges = (
                0.010951, 0.090349, 0.238193, 0.443532, 0.722793, 1.070500,
                1.612594, 2.409309, 3.841205, 7.000684, 30.997947
            )
        loss_fn = LogNormalBasisHazardLoss(centers=list(bin_edges))
        n_dim = len(bin_edges)

    loss_fn.to(args.device)

    # Initialize Model Structure
    # Assuming 'dataset' parameters are needed for init, usually stored in config
    # or we infer from a dummy dataset load.
    # For robust loading, we load dataset first.
    data_prefix = cfg.get("data_prefix", "ukb")
    cov_list = None if cfg.get("full_cov") else ["bmi", "smoking", "alcohol"]

    dataset = HealthDataset(data_prefix=data_prefix, covariate_list=cov_list)

    model_type = cfg.get("model_type", "delphifork")
    n_embd = cfg.get("n_embd", 120)
    pdrop = cfg.get("pdrop", 0.0)

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
            pdrop=pdrop,
            age_encoder_type=cfg.get("age_encoder", "sinusoidal"),
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
            n_layer=cfg.get("n_layer", 12),
            n_head=cfg.get("n_head", 12),
            pdrop=pdrop,
            age_encoder_type=cfg.get("age_encoder", "sinusoidal"),
            n_dim=n_dim,
            pretrained_weights_path=cfg.get("pretrained_weights_path"),
        )

    model.to(args.device)

    # Load Weights
    ckpt_path = os.path.join(args.run_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args.run_dir, "last_model.pt")

    print(f"Loading weights: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(state['model_state_dict'])

    # Also load criterion state if available (for log_sigma in LogNormal)
    if 'criterion_state_dict' in state and loss_fn is not None:
        loss_fn.load_state_dict(state['criterion_state_dict'])
        print("Loaded criterion state (sigma).")

    model.eval()

    # Split Test Set
    n_total = len(dataset)
    train_ratio = cfg.get("train_ratio", 0.7)
    val_ratio = cfg.get("val_ratio", 0.15)
    test_ratio = 1.0 - train_ratio - val_ratio

    tr_len = int(n_total * train_ratio)
    va_len = int(n_total * val_ratio)
    te_len = n_total - tr_len - va_len

    _, _, test_dataset = random_split(
        dataset, [tr_len, va_len, te_len],
        generator=torch.Generator().manual_seed(cfg.get("random_seed", 42))
    )

    disease_map = load_labels()

    # Run Experiments
    run_stratified_evaluation(test_dataset, model, loss_fn, args, disease_map)
    run_landmark_analysis(test_dataset, model, loss_fn, args)

    print("\nAll Evaluations Completed Successfully.")
