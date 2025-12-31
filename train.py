import json
import os
import time
import argparse
import math
from dataclasses import asdict, dataclass
from typing import Literal, Sequence
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from dataset import HealthDataset, health_collate_fn
from model import DelphiFork
from losses import (
    ExponentialNLLLoss,
    WeibullNLLLoss,
    LogNormalBasisHazardLoss,
    get_valid_pairs_and_dt,
)


@dataclass
class TrainConfig:
    """
    Configuration for training the DelphiFork model.

    Args:
        loss_type (str): Type of loss function. Options: 'exponential', 'weibull', 'lognormal'.
        age_encoder (str): Age encoder type. Options: 'sinusoidal', 'mlp'.
        full_cov (bool): Whether to use full covariance matrix.
        n_embd (int): Embedding dimension.
        n_layer (int): Number of Transformer layers.
        n_head (int): Number of attention heads.
        pdrop (float): Dropout probability.
        lambda_reg (float): Regularization weight.
        data_prefix (str): Data prefix.
        train_ratio (float): Training set ratio.
        val_ratio (float): Validation set ratio.
        test_ratio (float): Test set ratio.
        random_seed (int): Random seed.
        batch_size (int): Batch size.
        max_epochs (int): Maximum number of epochs.
        warmup_frac (float): Warmup fraction of total optimization steps.
        patient_epochs (int): Patience epochs for early stopping.
        min_epochs (int): Minimum number of epochs before early stopping can trigger.
        min_lr (float): Minimum learning rate.
        max_lr (float): Maximum learning rate.
        weight_decay (float): Weight decay for optimizer.
        grad_clip (float): Gradient clipping value.
        device (str): Device to use for training.
        trimmed_mean_trim (float): Trim fraction for trimmed-mean validation NLL.
    """
    # Model parameters
    # Options: 'exponential', 'weibull', 'lognormal'
    loss_type: Literal[
        "exponential",
        "weibull",
        "lognormal",
    ] = "weibull"
    age_encoder: Literal["sinusoidal", "mlp"] = "sinusoidal"
    full_cov: bool = False
    n_embd: int = 120
    n_layer: int = 12
    n_head: int = 12
    pdrop: float = 0.0
    lambda_reg: float = 1e-4
    # Data parameters
    data_prefix: str = "ukb"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    # Training parameters
    batch_size: int = 128
    max_epochs: int = 200
    warmup_frac: float = 0.1
    patient_epochs: int = 10
    min_epochs: int = 10
    min_lr: float = 6e-5
    max_lr: float = 6e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Metrics / validation parameters
    trimmed_mean_trim: float = 0.01


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train DelphiFork model for time-to-event prediction"
    )
    parser.add_argument(
        "--resume_run",
        type=str,
        default=None,
        help="Resume training from an existing run directory under runs/",
    )
    parser.add_argument("--loss_type", type=str,
                        default="weibull", help="Type of loss function")
    parser.add_argument("--full_cov", action="store_true",
                        help="Use full covariance matrix")
    parser.add_argument("--age_encoder", type=str,
                        default="sinusoidal", help="Age encoder type")
    parser.add_argument("--n_embd", type=int,
                        default=120, help="Embedding dimension")
    parser.add_argument("--n_layer", type=int,
                        default=12, help="Number of Transformer layers")
    parser.add_argument("--n_head", type=int,
                        default=12, help="Number of attention heads")
    parser.add_argument("--pdrop", type=float,
                        default=0.0, help="Dropout probability")
    parser.add_argument("--lambda_reg", type=float,
                        default=1e-4, help="Regularization weight")
    parser.add_argument("--data_prefix", type=str,
                        default="ukb", help="Data prefix")
    parser.add_argument("--batch_size", type=int,
                        default=128, help="Batch size")
    parser.add_argument("--max_epochs", type=int,
                        default=200, help="Maximum number of epochs")
    parser.add_argument("--warmup_frac", type=float,
                        default=0.1, help="Warmup fraction of total optimization steps (e.g., 0.1 = 10%%)")
    parser.add_argument("--patient_epochs", type=int,
                        default=10, help="Number of patient epochs for early stopping")
    parser.add_argument("--min_epochs", type=int,
                        default=10, help="Minimum number of epochs before early stopping")
    parser.add_argument("--min_lr", type=float,
                        default=6e-5, help="Minimum learning rate")
    parser.add_argument("--max_lr", type=float,
                        default=6e-4, help="Maximum learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=1e-2, help="Weight decay for optimizer")
    parser.add_argument("--grad_clip", type=float,
                        default=1.0, help="Gradient clipping value")
    parser.add_argument(
        "--trimmed_mean_trim",
        type=float,
        default=0.01,
        help="Trim fraction for trimmed-mean validation NLL (e.g., 0.1 trims 10%% on each side)",
    )
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    args = parser.parse_args()

    # Resume run: load config from run dir; ignore other CLI knobs (except device).
    if args.resume_run is not None:
        run_dir = Path(args.resume_run)
        config_path = run_dir / "train_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"resume_run missing train_config.json: {config_path}")
        with open(config_path, "r") as f:
            cfg_dict = json.load(f)
        cfg = TrainConfig(**cfg_dict)
        cfg.device = args.device
        # Attach resume path for __main__ to consume.
        cfg._resume_run = str(run_dir)  # type: ignore[attr-defined]
        return cfg

    config_dict = vars(args)
    config_dict.pop("resume_run", None)
    return TrainConfig(**config_dict)


def get_num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


bin_edges: Sequence[float] = (
    0.010951,  # ~ 4  days (p02)
    0.090349,  # ~ 33 days (p10)
    0.238193,  # ~ 87 days (p20)
    0.443532,  # ~ 162 days (p30)
    0.722793,  # ~ 264 days (p40)
    1.070500,  # ~ 391 days (p50)
    1.612594,  # ~ 589 days (p60)
    2.409309,  # ~ 880 days (p70)
    3.841205,  # ~ 1403 days (p80)
    7.000684,  # ~ 2557 days (p90)
    30.997947,  # ~ 11322 days (p99)
)


class Trainer:
    """
    Trainer class for the DelphiFork model.

    Args:
        cfg (TrainConfig): Training configuration.
    """

    def __init__(
        self,
        cfg: TrainConfig,
    ):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.start_epoch = 0
        self.global_step = 0

        if cfg.full_cov:
            cov_list = None
        else:
            cov_list = ["bmi", "smoking", "alcohol"]
        dataset = HealthDataset(
            data_prefix=cfg.data_prefix,
            covariate_list=cov_list,
        )
        print("Dataset loaded.")
        n_total = len(dataset)
        print(f"Total samples in dataset: {n_total}")
        print(f"Number of diseases: {dataset.n_disease}")
        print(f"Number of continuous covariates: {dataset.n_cont}")
        print(f"Number of categorical covariates: {dataset.n_cate}")
        self.train_data, self.val_data, _ = random_split(
            dataset,
            [
                int(n_total * cfg.train_ratio),
                int(n_total * cfg.val_ratio),
                n_total - int(n_total * cfg.train_ratio) -
                int(n_total * cfg.val_ratio),
            ],
            generator=torch.Generator().manual_seed(cfg.random_seed),
        )

        # Initial loaders (train loader will be rebuilt at stage boundaries).
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=health_collate_fn,
        )
        self.val_loader = DataLoader(
            self.val_data,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=health_collate_fn,
        )

        if cfg.loss_type == "exponential":
            self.criterion = ExponentialNLLLoss(
                lambda_reg=cfg.lambda_reg,
            ).to(self.device)
            n_dim = 1
        elif cfg.loss_type == "weibull":
            self.criterion = WeibullNLLLoss(
                lambda_reg=cfg.lambda_reg,
            ).to(self.device)
            n_dim = 2
        elif cfg.loss_type == "lognormal":
            centers = list(bin_edges)
            self.criterion = LogNormalBasisHazardLoss(
                centers=centers,
                bandwidth_scale=1.0,
                lambda_reg=cfg.lambda_reg,
            ).to(self.device)
            n_dim = len(centers)
        else:
            raise ValueError(f"Invalid loss type: {cfg.loss_type}")

        self.model = DelphiFork(
            n_disease=dataset.n_disease,
            n_tech_tokens=2,
            n_cont=dataset.n_cont,
            n_cate=dataset.n_cate,
            cate_dims=dataset.cate_dims,
            n_embd=cfg.n_embd,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            pdrop=cfg.pdrop,
            age_encoder_type=cfg.age_encoder,
            n_dim=n_dim,
        ).to(self.device)

        n_params = get_num_params(self.model)
        print(f"Model initialized. Number of parameters: {n_params}")
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.max_lr,
            weight_decay=cfg.weight_decay,
        )
        self.total_steps = (
            len(self.train_loader) * cfg.max_epochs
        )
        print(f"Total optimization steps: {self.total_steps}")

        resume_dir = getattr(cfg, "_resume_run", None)
        if resume_dir is not None:
            self.out_dir = str(resume_dir)
            os.makedirs(self.out_dir, exist_ok=True)
        else:
            while True:
                cov_suffix = "fullcov" if cfg.full_cov else "nocov"
                name = f"{cfg.loss_type}_{cfg.age_encoder}_{cov_suffix}"
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                model_dir = os.path.join("runs", f"{name}_{timestamp}")
                if not os.path.exists(model_dir):
                    self.out_dir = model_dir
                    os.makedirs(model_dir)
                    break
                time.sleep(2)
            self.save_config()

        print(f"Output directory: {self.out_dir}")
        self.best_path = os.path.join(self.out_dir, "best_model.pt")
        self.last_path = os.path.join(self.out_dir, "last_model.pt")

        self.start_epoch = 0
        self.global_step = 0

        if resume_dir is not None and os.path.exists(self.last_path):
            ckpt = torch.load(self.last_path, map_location=self.device)
            if isinstance(ckpt, dict):
                if "model_state_dict" in ckpt:
                    self.model.load_state_dict(
                        ckpt["model_state_dict"], strict=True)
                if "optimizer_state_dict" in ckpt:
                    self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                self.start_epoch = int(ckpt.get("epoch", -1)) + 1
                self.global_step = int(ckpt.get("global_step", 0))
                print(
                    f"Resumed from checkpoint: epoch={self.start_epoch}, global_step={self.global_step}"
                )

    def save_config(self):
        cfg_path = os.path.join(self.out_dir, "train_config.json")
        with open(cfg_path, "w") as f:
            json.dump(asdict(self.cfg), f, indent=4)
        print(f"Training configuration saved to {cfg_path}")

    def compute_lr(self, current_step: int) -> float:
        """Compute learning rate with linear warmup and cosine decay."""
        total_steps = max(int(self.total_steps), 1)
        warmup_steps = int(self.cfg.warmup_frac * total_steps)
        warmup_steps = max(warmup_steps, 1)
        if current_step < warmup_steps:
            lr = self.cfg.max_lr * (current_step / warmup_steps)
        else:
            denom = max(total_steps - warmup_steps, 1)
            progress = (current_step - warmup_steps) / denom
            progress = min(max(progress, 0.0), 1.0)
            lr = self.cfg.min_lr + 0.5 * (
                self.cfg.max_lr - self.cfg.min_lr
            ) * (1 + math.cos(math.pi * progress))
        return lr

    def compute_trimmed_mean(self, values: torch.Tensor, trim: float) -> float:
        """Compute trimmed mean of a tensor."""
        if values.numel() == 0:
            return 0.0
        sorted_vals, _ = torch.sort(values)
        n = sorted_vals.size(0)
        k = int(n * trim)
        if n - 2 * k <= 0:
            return sorted_vals.float().mean().item()
        trimmed_vals = sorted_vals[k:n - k]
        return trimmed_vals.float().mean().item()

    def train(self) -> None:
        # If resuming, keep loaded global_step.
        history = []
        best_val_score = float("inf")
        patient_counter = 0
        for epoch in tqdm(range(self.start_epoch, self.cfg.max_epochs), desc="Overall Training"):
            self.model.train()
            running_nll = 0.0
            running_reg = 0.0

            batch_count = 0
            pbar = tqdm(self.train_loader,
                        desc=f"Epoch {epoch+1}/{self.cfg.max_epochs}")
            for batch in pbar:
                (
                    event_seq,
                    time_seq,
                    cont_feats,
                    cate_feats,
                    sexes,
                ) = batch
                event_seq = event_seq.to(self.device)
                time_seq = time_seq.to(self.device)
                cont_feats = cont_feats.to(self.device)
                cate_feats = cate_feats.to(self.device)
                sexes = sexes.to(self.device)
                res = get_valid_pairs_and_dt(event_seq, time_seq, 2)
                if res is None:
                    continue
                dt, b_prev, t_prev, b_next, t_next = res
                self.optimizer.zero_grad()
                lr = self.compute_lr(self.global_step)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
                logits = self.model(
                    event_seq,
                    time_seq,
                    sexes,
                    cont_feats,
                    cate_feats,
                    b_prev=b_prev,
                    t_prev=t_prev
                )
                target_events = event_seq[b_next, t_next] - 2
                nll_vec, reg = self.criterion(
                    logits,
                    target_events,
                    dt,
                    reduction="none",
                )
                finite_mask = torch.isfinite(nll_vec)
                if not finite_mask.any():
                    continue
                nll_vec = nll_vec[finite_mask]
                nll = nll_vec.mean()

                loss = nll + reg
                batch_count += 1
                running_nll += nll.item()
                running_reg += reg.item()
                pbar.set_postfix({
                    "lr": lr,
                    "NLL": running_nll / batch_count,
                    "Reg": running_reg / batch_count,
                })
                loss.backward()
                if self.cfg.grad_clip > 0.0:
                    clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()
                self.global_step += 1

            if batch_count == 0:
                print("No valid training pairs found this epoch; skipping epoch.")
                continue

            train_nll = running_nll / batch_count
            train_reg = running_reg / batch_count
            self.model.eval()
            val_batch_count = 0
            total_val_pairs = 0
            total_val_nll = 0.0
            total_val_reg = 0.0
            val_nll_list = []
            val_dt_list = []
            with torch.no_grad():
                val_pbar = tqdm(self.val_loader, desc="Validation")
                for batch in val_pbar:
                    (
                        event_seq,
                        time_seq,
                        cont_feats,
                        cate_feats,
                        sexes,
                    ) = batch
                    event_seq = event_seq.to(self.device)
                    time_seq = time_seq.to(self.device)
                    cont_feats = cont_feats.to(self.device)
                    cate_feats = cate_feats.to(self.device)
                    sexes = sexes.to(self.device)
                    res = get_valid_pairs_and_dt(event_seq, time_seq, 2)
                    if res is None:
                        continue
                    dt, b_prev, t_prev, b_next, t_next = res
                    num_pairs = dt.size(0)
                    logits = self.model(
                        event_seq,
                        time_seq,
                        sexes,
                        cont_feats,
                        cate_feats,
                        b_prev=b_prev,
                        t_prev=t_prev
                    )
                    target_events = event_seq[b_next, t_next] - 2
                    nll, reg = self.criterion(
                        logits,
                        target_events,
                        dt,
                        reduction="none",
                    )
                    val_nll_list.append(nll.cpu())
                    val_dt_list.append(dt.cpu())
                    batch_nll_sum = nll.sum().item()
                    total_val_nll += batch_nll_sum
                    total_val_reg += reg.item() * num_pairs
                    total_val_pairs += num_pairs

                    current_val_avg_nll = total_val_nll / \
                        total_val_pairs if total_val_pairs > 0 else 0.0
                    current_val_avg_reg = total_val_reg / \
                        total_val_pairs if total_val_pairs > 0 else 0.0

                    val_pbar.set_postfix({
                        "NLL": f"{current_val_avg_nll:.4f}",
                        "Reg": f"{current_val_avg_reg:.4f}",
                    })
                    val_batch_count += 1

            val_nll = total_val_nll / \
                total_val_pairs if total_val_pairs > 0 else 0.0
            val_reg = total_val_reg / \
                total_val_pairs if total_val_pairs > 0 else 0.0
            if len(val_nll_list) > 0:
                val_nll_tensor = torch.cat(val_nll_list)
                trimmed_mean_nll = self.compute_trimmed_mean(
                    val_nll_tensor,
                    trim=min(max(float(self.cfg.trimmed_mean_trim), 0.0), 0.49),
                )
            else:
                val_nll_tensor = torch.tensor([])
                trimmed_mean_nll = 0.0

            if len(val_dt_list) > 0:
                val_dt_tensor = torch.cat(val_dt_list)
            else:
                val_dt_tensor = torch.tensor([])

            median_val_nll = torch.median(
                val_nll_tensor).item() if len(val_nll_tensor) > 0 else 0.0
            p90_val_nll = torch.quantile(val_nll_tensor, 0.90).item() if len(
                val_nll_tensor) > 0 else 0.0
            p95_val_nll = torch.quantile(val_nll_tensor, 0.95).item() if len(
                val_nll_tensor) > 0 else 0.0
            p99_val_nll = torch.quantile(val_nll_tensor, 0.99).item() if len(
                val_nll_tensor) > 0 else 0.0
            nonfinite_count = torch.sum(~torch.isfinite(val_nll_tensor)).item()
            nonfinite_rate = nonfinite_count / \
                len(val_nll_tensor) if len(val_nll_tensor) > 0 else 0.0

            # Bucketed stats
            # Buckets: [0, p50], (p50, p90], (p90, p99], (p99, inf)
            # Thresholds are hard-coded (and correspond to values in bin_edges).
            p50 = 1.070500
            p90 = 7.000684
            p99 = 30.997947

            buckets = {
                "p00-p50": (val_dt_tensor <= p50),
                "p50-p90": (val_dt_tensor > p50) & (val_dt_tensor <= p90),
                "p90-p99": (val_dt_tensor > p90) & (val_dt_tensor <= p99),
                "p99-inf": (val_dt_tensor > p99),
            }

            bucket_stats = {}
            for name, mask in buckets.items():
                if mask.any():
                    bnll = val_nll_tensor[mask]
                    bucket_stats[name] = {
                        "count": int(bnll.numel()),
                        "mean": bnll.mean().item(),
                        "median": torch.median(bnll).item(),
                        "nonfinite": torch.sum(~torch.isfinite(bnll)).item() / len(bnll)
                    }
                else:
                    bucket_stats[name] = {
                        "count": 0, "mean": 0.0, "median": 0.0, "nonfinite": 0.0}

            history.append({
                "epoch": epoch,
                "train_nll": train_nll,
                "train_reg": train_reg,
                "val_nll": val_nll,
                "val_reg": val_reg,
                "val_nll_median": median_val_nll,
                "val_nll_p90": p90_val_nll,
                "val_nll_p95": p95_val_nll,
                "val_nll_p99": p99_val_nll,
                "val_nonfinite_rate": nonfinite_rate,
                "val_nll_trimmed_mean": trimmed_mean_nll,
                "bucket_stats": bucket_stats,
            })

            print(f"Epoch {epoch} Stats:")
            print(f"  Train NLL: {train_nll:.4f}")
            print(f"  Val NLL (Mean): {val_nll:.4f}")
            print(f"  Val NLL (Median): {median_val_nll:.4f}")
            print(
                f"  Val NLL (Trimmed Mean): {trimmed_mean_nll:.4f} ← PRIMARY METRIC")
            print(f"  Val NLL (P90): {p90_val_nll:.4f}")
            print(f"  Val NLL (P95): {p95_val_nll:.4f}")
            print(f"  Val NLL (P99): {p99_val_nll:.4f}")
            print(f"  Val Non-Finite Rate: {nonfinite_rate:.4f}")
            print("  Bucket Stats (Mean/Median/NonFinite):")
            for name, stats in bucket_stats.items():
                print(
                    f"    {name}: {stats['mean']:.4f} / {stats['median']:.4f} / {stats['nonfinite']:.4f} (n={stats.get('count', 0)})")

            with open(os.path.join(self.out_dir, "training_history.json"), "w") as f:
                json.dump(history, f, indent=4)

            # Check for improvement
            if trimmed_mean_nll < best_val_score:
                best_val_score = trimmed_mean_nll
                patient_counter = 0
                print("New best validation score. Saving checkpoint.")

                torch.save({
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }, self.best_path)
            else:
                patient_counter += 1
                if epoch+1 >= self.cfg.min_epochs and patient_counter >= self.cfg.patient_epochs:
                    print(
                        f"No improvement in validation score for {patient_counter} epochs. Early stopping.")
                    return
                print("No improvement in validation score.")

            torch.save({
                "epoch": epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }, self.last_path)

        print("Training complete.")


if __name__ == "__main__":
    cfg = parse_args()
    trainer = Trainer(cfg)
    trainer.train()
