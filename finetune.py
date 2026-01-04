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
from model import DelphiFork, SapDelphi
from losses import (
    ExponentialNLLLoss,
    LogNormalBasisHazardLoss,
    get_valid_pairs_and_dt,
)


@dataclass
class TrainConfig:
    """
    Configuration for fine-tuning the DelphiFork model.

    Args:
        loss_type (str): Type of loss function. Options: 'lognormal'.
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
        finetune_from (str): Path to the checkpoint .pt file to fine-tune from.
    """
    # Model parameters
    model_type: Literal["sapdelphi"] = "sapdelphi"
    # Options: 'lognormal'
    loss_type: Literal[
        "lognormal",
    ] = "lognormal"
    age_encoder: Literal["sinusoidal", "mlp"] = "sinusoidal"
    full_cov: bool = False
    n_embd: int = 120
    n_layer: int = 12
    n_head: int = 12
    pdrop: float = 0.2  # Changed to 0.2 for fine-tuning
    lambda_reg: float = 1e-4
    # SapDelphi-specific parameters
    pretrained_weights_path: str = "icd10_sapbert_embeddings.npy"
    freeze_embeddings: bool = False  # Forced to False in Trainer
    # Data parameters
    data_prefix: str = "ukb"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    # Training parameters
    batch_size: int = 128
    max_epochs: int = 200
    warmup_frac: float = 0.05  # Changed to 0.05 for fine-tuning
    patient_epochs: int = 10
    min_epochs: int = 10
    min_lr: float = 5e-6  # Changed to 5e-6 for fine-tuning
    max_lr: float = 5e-5  # Changed to 5e-5 for fine-tuning
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # EMA parameters
    ema_decay: float = 0.999
    # Fine-tuning parameters
    finetune_from: str = ""  # Required


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Fine-tune DelphiFork model for time-to-event prediction"
    )
    parser.add_argument(
        "--finetune_from",
        type=str,
        required=True,
        help="Path to the checkpoint .pt file to fine-tune from",
    )
    parser.add_argument("--model_type", type=str,
                        default="sapdelphi", help="Model type: sapdelphi")
    parser.add_argument("--loss_type", type=str,
                        default="lognormal", help="Type of loss function")
    parser.add_argument("--full_cov", action="store_true",
                        help="Use full covariance matrix")
    parser.add_argument("--age_encoder", type=str,
                        default="sinusoidal", help="Age encoder type")
    parser.add_argument("--pretrained_weights_path", type=str,
                        default="icd10_sapbert_embeddings.npy", help="Path to pretrained embeddings (SapDelphi only)")
    parser.add_argument("--freeze_embeddings", action="store_true",
                        help="Freeze pretrained embeddings (Ignored in fine-tuning)")
    parser.add_argument("--n_embd", type=int,
                        default=120, help="Embedding dimension")
    parser.add_argument("--n_layer", type=int,
                        default=12, help="Number of Transformer layers")
    parser.add_argument("--n_head", type=int,
                        default=12, help="Number of attention heads")
    parser.add_argument("--pdrop", type=float,
                        default=0.2, help="Dropout probability")
    parser.add_argument("--lambda_reg", type=float,
                        default=1e-4, help="Regularization weight")
    parser.add_argument("--data_prefix", type=str,
                        default="ukb", help="Data prefix")
    parser.add_argument("--batch_size", type=int,
                        default=128, help="Batch size")
    parser.add_argument("--max_epochs", type=int,
                        default=200, help="Maximum number of epochs")
    parser.add_argument("--warmup_frac", type=float,
                        default=0.05, help="Warmup fraction of total optimization steps")
    parser.add_argument("--patient_epochs", type=int,
                        default=10, help="Number of patient epochs for early stopping")
    parser.add_argument("--min_epochs", type=int,
                        default=10, help="Minimum number of epochs before early stopping")
    parser.add_argument("--min_lr", type=float,
                        default=5e-6, help="Minimum learning rate")
    parser.add_argument("--max_lr", type=float,
                        default=5e-5, help="Maximum learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=1e-2, help="Weight decay for optimizer")
    parser.add_argument("--grad_clip", type=float,
                        default=1.0, help="Gradient clipping value")
    parser.add_argument("--ema_decay", type=float,
                        default=0.9999, help="EMA decay rate for model parameters")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    args = parser.parse_args()

    if args.model_type != "sapdelphi":
        raise ValueError(
            "Fine-tuning is only supported for 'sapdelphi' model type.")
    if args.loss_type not in ["lognormal"]:
        raise ValueError(
            "Fine-tuning is only supported for 'lognormal' loss types.")

    config_dict = vars(args)
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
    Trainer class for fine-tuning the DelphiFork model.

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

        if cfg.loss_type == "lognormal":
            centers = list(bin_edges)
            self.criterion = LogNormalBasisHazardLoss(
                centers=centers,
                bandwidth_scale=1.0,
                lambda_reg=cfg.lambda_reg,
            ).to(self.device)
            n_dim = len(centers)
        else:
            raise ValueError(
                f"Invalid loss type for fine-tuning: {cfg.loss_type}. Supported: lognormal")

        if cfg.model_type == "sapdelphi":
            # FORCE freeze_embeddings=False for fine-tuning
            self.model = SapDelphi(
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
                pretrained_weights_path=cfg.pretrained_weights_path,
                freeze_embeddings=False,  # Forced to False
            ).to(self.device)
        else:
            raise ValueError(
                f"Invalid model_type for fine-tuning: {cfg.model_type}. Supported: sapdelphi")

        # Load checkpoint
        print(f"Loading checkpoint from {cfg.finetune_from}...")
        ckpt = torch.load(cfg.finetune_from, map_location=self.device)

        # Load model weights
        if "model_state_dict" in ckpt:
            # Use strict=False to allow flexibility
            keys = self.model.load_state_dict(
                ckpt["model_state_dict"], strict=False)
            print(
                f"Model weights loaded. Missing keys: {keys.missing_keys}, Unexpected keys: {keys.unexpected_keys}")
        else:
            print("Warning: 'model_state_dict' not found in checkpoint.")

        # Load criterion parameters if present (e.g., LogNormalBasisHazardLoss.log_sigma)
        if "criterion_state_dict" in ckpt:
            try:
                self.criterion.load_state_dict(
                    ckpt["criterion_state_dict"], strict=True)
                print("Criterion state loaded from checkpoint.")
            except Exception as e:
                print(
                    f"Warning: failed to load criterion_state_dict (continuing with fresh criterion params): {e}")

        n_params = get_num_params(self.model)
        print(f"Model initialized. Number of parameters: {n_params}")

        # Initialize EMA model
        self.ema_model = self._create_ema_model()

        # Set up optimizer with differential learning rates for SapDelphi
        if cfg.model_type == "sapdelphi":
            # token_embedding gets 0.1x the main learning rate
            emb_params = list(self.model.token_embedding.parameters())
            other_params = [
                p for n, p in self.model.named_parameters()
                if "token_embedding" not in n
            ]
            # Include criterion parameters (e.g., LogNormalBasisHazardLoss.log_sigma)
            other_params.extend(list(self.criterion.parameters()))
            self.optimizer = AdamW(
                [
                    {"params": emb_params, "lr": cfg.max_lr * 0.1},
                    {"params": other_params, "lr": cfg.max_lr},
                ],
                weight_decay=cfg.weight_decay,
            )
            self._has_differential_lr = True
        else:
            self.optimizer = AdamW(
                list(self.model.parameters()) +
                list(self.criterion.parameters()),
                lr=cfg.max_lr,
                weight_decay=cfg.weight_decay,
            )
            self._has_differential_lr = False

        self.total_steps = (
            len(self.train_loader) * cfg.max_epochs
        )
        print(f"Total optimization steps: {self.total_steps}")

        # Output directory setup
        while True:
            cov_suffix = "fullcov" if cfg.full_cov else "nocov"
            name = f"finetune_{cfg.model_type}_{cfg.loss_type}_{cfg.age_encoder}_{cov_suffix}"
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

        # Reset state
        self.start_epoch = 0
        self.global_step = 0
        # Do NOT load optimizer or scheduler state.

    def save_config(self):
        cfg_path = os.path.join(self.out_dir, "train_config.json")
        with open(cfg_path, "w") as f:
            json.dump(asdict(self.cfg), f, indent=4)
        print(f"Training configuration saved to {cfg_path}")

    def _create_ema_model(self):
        """Create a copy of the model for EMA."""
        # Simply clone the model structure and load current weights
        if self.cfg.model_type == "sapdelphi":
            ema_model = SapDelphi(
                n_disease=self.model.n_disease,
                n_tech_tokens=self.model.n_tech_tokens,
                n_cont=self.model.tabular_encoder.n_cont,
                n_cate=self.model.tabular_encoder.n_cate,
                cate_dims=[] if self.model.tabular_encoder.n_cate == 0 else [
                    emb.num_embeddings for emb in self.model.tabular_encoder.cate_embds
                ],
                n_embd=self.model.n_embd,
                n_layer=len(self.model.blocks),
                n_head=self.model.n_head,
                pdrop=self.cfg.pdrop,
                age_encoder_type=self.cfg.age_encoder,
                n_dim=self.model.n_dim,
                pretrained_weights_path=self.cfg.pretrained_weights_path,
                freeze_embeddings=False,  # Forced to False
            )
        else:
            raise ValueError(
                f"Invalid model_type for fine-tuning: {self.cfg.model_type}. Supported: sapdelphi")

        ema_model.load_state_dict(self.model.state_dict())
        ema_model.to(self.device)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model

    def _update_ema(self):
        """Update EMA model parameters with adaptive decay."""
        # Adaptive EMA decay
        current_decay = min(
            self.cfg.ema_decay,
            (1 + self.global_step) / (10 + self.global_step)
        )

        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(current_decay).add_(
                    model_param.data, alpha=1 - current_decay)

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

    def train(self) -> None:
        # If resuming, keep loaded global_step.
        history = []
        best_val_score = float("inf")
        patient_counter = 0
        for epoch in range(self.start_epoch, self.cfg.max_epochs):
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
                if self._has_differential_lr:
                    # Apply 0.1x for token_embedding, 1.0x for others
                    self.optimizer.param_groups[0]["lr"] = lr * 0.1
                    self.optimizer.param_groups[1]["lr"] = lr
                else:
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
                    clip_grad_norm_(
                        list(self.model.parameters()) +
                        list(self.criterion.parameters()),
                        self.cfg.grad_clip,
                    )
                self.optimizer.step()
                self._update_ema()
                self.global_step += 1

            if batch_count == 0:
                tqdm.write(
                    "No valid training pairs found this epoch; skipping epoch.")
                continue

            train_nll = running_nll / batch_count
            train_reg = running_reg / batch_count

            # Validation with EMA model
            self.ema_model.eval()
            total_val_pairs = 0
            total_val_nll = 0.0
            total_val_reg = 0.0
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
                    logits = self.ema_model(
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

            val_nll = total_val_nll / total_val_pairs if total_val_pairs > 0 else 0.0
            val_reg = total_val_reg / total_val_pairs if total_val_pairs > 0 else 0.0

            history.append({
                "epoch": epoch,
                "train_nll": train_nll,
                "train_reg": train_reg,
                "val_nll": val_nll,
                "val_reg": val_reg,
            })

            tqdm.write(f"\nEpoch {epoch+1}/{self.cfg.max_epochs} Stats:")
            tqdm.write(f"  Train NLL: {train_nll:.4f}")
            tqdm.write(f"  Val NLL: {val_nll:.4f} ← PRIMARY METRIC")

            with open(os.path.join(self.out_dir, "training_history.json"), "w") as f:
                json.dump(history, f, indent=4)

            # Check for improvement
            if val_nll < best_val_score:
                best_val_score = val_nll
                patient_counter = 0
                tqdm.write("  ✓ New best validation score. Saving checkpoint.")

                torch.save({
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "model_state_dict": self.ema_model.state_dict(),
                    "criterion_state_dict": self.criterion.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }, self.best_path)
            else:
                patient_counter += 1
                if epoch+1 >= self.cfg.min_epochs and patient_counter >= self.cfg.patient_epochs:
                    tqdm.write(
                        f"\n⚠ No improvement in validation score for {patient_counter} epochs. Early stopping.")
                    return
                tqdm.write(
                    f"  No improvement (patience: {patient_counter}/{self.cfg.patient_epochs})")

            torch.save({
                "epoch": epoch,
                "global_step": self.global_step,
                "model_state_dict": self.ema_model.state_dict(),
                "criterion_state_dict": self.criterion.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }, self.last_path)

        tqdm.write("\n🎉 Training complete!")


if __name__ == "__main__":
    cfg = parse_args()
    trainer = Trainer(cfg)
    trainer.train()
