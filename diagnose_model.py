
from evaluate_final import load_labels
from losses import LogNormalBasisHazardLoss, ExponentialNLLLoss
from dataset import HealthDataset, health_collate_fn
from model import DelphiFork, SapDelphi
import os
import argparse
import sys
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# Add current directory to path
sys.path.append(os.getcwd())

# Import from existing modules


def get_validation_set(cfg_dict, dataset):
    """Reconstruct validation split based on config."""
    n_total = len(dataset)
    train_ratio = cfg_dict.get("train_ratio", 0.7)
    val_ratio = cfg_dict.get("val_ratio", 0.15)

    train_len = int(n_total * train_ratio)
    val_len = int(n_total * val_ratio)
    test_len = n_total - train_len - val_len

    random_seed = cfg_dict.get("random_seed", 42)

    _, val_dataset, _ = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(random_seed)
    )
    return val_dataset


def compute_hazard_at_t(theta, t_val, loss_type, loss_fn):
    """
    Compute hazard h(t) for each disease at scalar time t_val (years).
    """
    device = theta.device
    B, K = theta.shape[0], theta.shape[1]

    if loss_type == "exponential":
        # theta is logits -> softplus -> lambda
        # h(t) = lambda (constant)
        if theta.dim() == 3:
            theta = theta.squeeze(-1)
        lambdas = F.softplus(theta)
        return lambdas  # (B, K)

    elif loss_type == "lognormal":
        # theta is coeffs (B, K, n_basis)
        # h(t) = exp( sum_b theta_kb * K_b(t) )
        if loss_fn is None:
            raise ValueError("loss_fn required for lognormal")

        t_tensor = torch.full((B,), t_val, device=device)
        K_vals = loss_fn._compute_kernel(t_tensor)  # (B, n_basis)

        # theta: (B, K, n_basis)
        # K_vals: (B, n_basis)
        # We need sum_b theta_kb * K_b

        # Expand K_vals to (B, 1, n_basis)
        K_exp = K_vals.unsqueeze(1)

        # log_h = sum(theta * K, dim=-1)
        log_h = torch.sum(theta * K_exp, dim=-1)  # (B, K)

        # Clamp for stability (similar to losses.py or evaluate_final.py)
        log_h = torch.clamp(log_h, max=20.0)

        return torch.exp(log_h)  # (B, K)

    return None


def diagnose(args):
    # 1. Setup
    config_path = os.path.join(args.run_dir, "train_config.json")
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return

    with open(config_path, "r") as f:
        cfg = json.load(f)
    print(f"Loaded config from {args.run_dir}")

    device = torch.device(args.device)

    # Load Dataset
    data_prefix = cfg.get("data_prefix", "ukb")
    if cfg.get("full_cov", False):
        cov_list = None
    else:
        cov_list = ["bmi", "smoking", "alcohol"]

    # Check if files exist
    if not os.path.exists(f"{data_prefix}_basic_info.csv"):
        print(f"Data files for {data_prefix} not found.")
        return

    dataset = HealthDataset(data_prefix=data_prefix, covariate_list=cov_list)
    val_dataset = get_validation_set(cfg, dataset)
    print(f"Validation set size: {len(val_dataset)}")

    # Load Labels
    labels_map = load_labels()

    # Initialize Loss & Model
    loss_type = args.loss_type if args.loss_type else cfg.get("loss_type")

    loss_fn = None
    n_dim = 1
    if loss_type == "exponential":
        loss_fn = ExponentialNLLLoss().to(device)
        n_dim = 1
    elif loss_type == "lognormal":
        bin_edges = (
            0.010951, 0.090349, 0.238193, 0.443532, 0.722793, 1.070500,
            1.612594, 2.409309, 3.841205, 7.000684, 30.997947
        )
        centers = list(bin_edges)
        n_dim = len(centers)
        loss_fn = LogNormalBasisHazardLoss(centers=centers).to(device)

    model_type = cfg.get("model_type", "delphifork")
    n_embd = cfg.get("n_embd", 120)

    # Initialize Model logic (copied/adapted from evaluate_final.py)
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
            pdrop=0.0,
            age_encoder_type=cfg.get("age_encoder", "sinusoidal"),
            n_dim=n_dim,
            pretrained_weights_path=cfg.get("pretrained_weights_path"),
        )

    model.to(device)

    # Load Checkpoint
    best_path = os.path.join(args.run_dir, "best_model.pt")
    if not os.path.exists(best_path):
        best_path = os.path.join(args.run_dir, "last_model.pt")

    print(f"Loading weights from {best_path}")
    ckpt = torch.load(best_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        if "criterion_state_dict" in ckpt and loss_fn is not None:
            loss_fn.load_state_dict(ckpt["criterion_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    # 2. Diagnosis Loop
    # We will simply predict 1 year out from the last event for every patient in validation

    loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=health_collate_fn
    )

    all_predicted_ids = []

    all_seq_lengths = []
    all_mean_risks = []
    all_coeffs = []  # Flattened theta

    horizon_years = 1.0  # T = 1 year ahead

    print("Running diagnosis inference...")
    with torch.no_grad():
        for batch in tqdm(loader):
            event_batch, time_batch, cont_batch, cate_batch, sex_batch = batch

            # Move inputs to device (EXCEPT event/time for filtering logic)
            # The model needs them on device, but we might pre-process on CPU if needed.
            # Here we just put everything on device directly like train.py.

            event_in = event_batch.to(device)
            time_in = time_batch.to(device)
            cont_in = cont_batch.to(device)
            cate_in = cate_batch.to(device)
            sex_in = sex_batch.to(device)

            B_curr = event_in.shape[0]

            # Build b_prev, t_prev for the *last* valid event of each sequence
            valid_lens = (event_in != 0).sum(dim=1)
            t_prev = valid_lens - 1
            b_prev = torch.arange(B_curr, device=device)

            # To avoid index -1 error for empty seqs (rare but possible in unfiltered data), clamp
            # But valid_lens should be >= 1 usually. If 0, skip.
            # Filtering empty
            has_data = valid_lens > 0
            if not has_data.all():
                event_in = event_in[has_data]
                time_in = time_in[has_data]
                cont_in = cont_in[has_data]
                cate_in = cate_in[has_data]
                sex_in = sex_in[has_data]
                b_prev = torch.arange(event_in.shape[0], device=device)
                t_prev = (event_in != 0).sum(dim=1) - 1

            if event_in.shape[0] == 0:
                continue

            theta = model(event_in, time_in, sex_in, cont_in,
                          cate_in, b_prev=b_prev, t_prev=t_prev)
            # theta shape (B, K, n_dim) if lognormal, or (B, K) if exp
            if loss_type == "lognormal":
                theta = theta.view(event_in.shape[0], dataset.n_disease, n_dim)

            # Collect Raw Coeffs (Suspect C: Distribution)
            # We take mean over basis dims or just flatten everything?
            # Prompt says "Histogram of raw log-hazard coefficients".
            # If lognormal, these are coeffs. If exp, logits.
            all_coeffs.append(theta.cpu().numpy().flatten())

            # Metric for Suspect C: Correlation
            # Seq Length
            seq_lens = (event_in != 0).sum(dim=1).cpu().numpy()
            all_seq_lengths.extend(seq_lens)

            # Mean Predicted Risk (Hazard at t=1)
            hazards = compute_hazard_at_t(
                theta, horizon_years, loss_type, loss_fn)
            # hazards: (B, K)

            # Mean risk across diseases for this patient
            mean_risk_patient = hazards.mean(dim=1).cpu().numpy()
            all_mean_risks.extend(mean_risk_patient)

            # Metric for Suspect B: Top-1 Token
            # argmax over diseases
            top_ids = torch.argmax(hazards, dim=1)  # (B,) values 0..n_disease-1
            # Map back to event ID (add 2)
            top_event_ids = (top_ids + 2).cpu().tolist()
            all_predicted_ids.extend(top_event_ids)

    # --- Analysis Report ---

    print("\n" + "="*50)
    print("DIAGNOSIS REPORT")
    print("="*50)

    # Suspect B: Junk Event Dominance
    print("\n>>> Suspect B: Top-1 Predicted Token Frequencies")
    id_counts = defaultdict(int)
    for eid in all_predicted_ids:
        id_counts[eid] += 1

    sorted_counts = sorted(id_counts.items(), key=lambda x: x[1], reverse=True)
    top_20 = sorted_counts[:20]

    print(f"{'Rank':<5} {'Event ID':<10} {'Name':<40} {'Count':<10} {'%':<5}")
    print("-" * 75)
    total_preds = len(all_predicted_ids)
    for i, (eid, count) in enumerate(top_20):
        name = labels_map.get(eid, "Unknown")
        pct = (count / total_preds) * 100
        print(f"{i+1:<5} {eid:<10} {name[:40]:<40} {count:<10} {pct:.1f}%")

    top_1_pct = top_20[0][1] / total_preds if top_20 else 0
    if top_1_pct > 0.5:
        print("\n[WARNING] Dominant token detected! Suspect B is likely confirmed.")

    # Suspect C: Distribution & Correlation
    print("\n>>> Suspect C: Survival Term Dominance & Length Correlation")

    all_coeffs_flat = np.concatenate(all_coeffs)
    print(
        f"Raw Coefficient Stats: Mean={all_coeffs_flat.mean():.4f}, Std={all_coeffs_flat.std():.4f}")
    if all_coeffs_flat.mean() < -5.0:
        print("[WARNING] Mean coefficient is very negative. Model might be predicting ~0 hazard everywhere.")

    arr_lens = np.array(all_seq_lengths)
    arr_risks = np.array(all_mean_risks)

    if len(arr_risks) > 1:
        corr = np.corrcoef(arr_lens, arr_risks)[0, 1]
    else:
        corr = 0.0

    print(f"Correlation (Seq Length vs Mean Risk): {corr:.4f}")

    if corr > 0.5:
        print(
            "[WARNING] High correlation (>0.5). Model might be using length as proxy for risk.")

    # --- Visualization ---

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Histogram of Coeffs
    axes[0].hist(all_coeffs_flat, bins=50, color='skyblue', edgecolor='black')
    axes[0].set_title("Histogram of Raw Coefficients (Theta)")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Scatter Length vs Risk
    axes[1].scatter(arr_lens, arr_risks, alpha=0.3, s=10)
    axes[1].set_title(f"Seq Length vs Mean Hazard (corr={corr:.2f})")
    axes[1].set_xlabel("Sequence Length")
    axes[1].set_ylabel("Mean Hazard (t=1yr)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("diagnosis_suspect_c.png")
    print("\nPlots saved to diagnosis_suspect_c.png")
    print("Diagnosis Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to training run directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--loss_type", type=str, default=None,
                        help="Override loss type if needed")

    args = parser.parse_args()

    diagnose(args)
