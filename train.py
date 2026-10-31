"""
train.py (nanoGentzen-v3 Kripke + Jev)
Trains / fine-tunes the GentzenPolicyValueModel with:
  1. Boundary-Aware Span Pointer Antecedent Pivot Attention Head (up to 32 antecedents)
  2. 3-Class Saul Kripke Epistemic Value Head: [P(False), P(Undecided), P(True)]
  3. Joint Cross-Entropy supervision over Rules, Pivots, and Epistemic Classes
  4. FP32 master weights with AMP (bf16/fp16 + GradScaler), decoupled weight decay,
     trajectory-grouped validation split, and best-checkpoint selection.
"""
import argparse
import copy
from dataclasses import asdict
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from torch.utils.data import DataLoader, Subset

try:
    from safetensors.torch import load_file, save_file
except ImportError:
    load_file, save_file = None, None

try:
    from nanogentzen.dataset import GentzenDataset
    from nanogentzen.modeling_nanogentzen import GentzenPolicyValueModel
    from nanogentzen.configuration_nanogentzen import GentzenConfig
    from nanogentzen.tokenization_nanogentzen import LogicTokenizer
except ImportError:
    from dataset import GentzenDataset
    from modeling_nanogentzen import GentzenPolicyValueModel
    from configuration_nanogentzen import GentzenConfig
    from tokenization_nanogentzen import LogicTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train nanoGentzen-v3 Saul Kripke Policy-Value Network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path", "--data_path", "--dataset-path", "--dataset_path",
        type=str,
        dest="dataset_path",
        default="data/gentzen_dataset.pt",
        help="Path to pre-generated .pt dataset",
    )
    parser.add_argument(
        "--config-path", "--config_path",
        type=str,
        dest="config_path",
        default="config.json",
        help="Path to architecture config.json",
    )
    parser.add_argument(
        "--resume-safetensors", "--resume_safetensors",
        type=str,
        dest="resume_safetensors",
        default="",
        help="Optional path to base model.safetensors to fine-tune from",
    )
    parser.add_argument(
        "--num-samples", "--num_samples",
        type=int,
        dest="num_samples",
        default=12000,
        help="Number of samples to synthesize on-demand if dataset does not exist",
    )
    parser.add_argument(
        "--val-split", "--val_split",
        type=float,
        dest="val_split",
        default=0.10,
        help="Fraction of trajectories held out for validation (0.0 to disable)",
    )
    parser.add_argument(
        "--batch-size", "--batch_size",
        type=int,
        dest="batch_size",
        default=128,
        help="Training batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr", "--learning-rate", "--learning_rate",
        type=float,
        dest="lr",
        default=5e-4,
        help="AdamW peak learning rate",
    )
    parser.add_argument(
        "--weight-decay", "--weight_decay",
        type=float,
        dest="weight_decay",
        default=0.01,
        help="Weight decay for 2D linear weights in AdamW",
    )
    parser.add_argument(
        "--warmup-ratio", "--warmup_ratio",
        type=float,
        dest="warmup_ratio",
        default=0.05,
        help="Fraction of total training steps used for linear LR warmup",
    )
    parser.add_argument(
        "--max-grad-norm", "--max_grad_norm",
        type=float,
        dest="max_grad_norm",
        default=1.0,
        help="Maximum gradient clipping norm",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device",
    )
    parser.add_argument(
        "--mixed-precision", "--mixed_precision",
        type=str,
        dest="mixed_precision",
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="Mixed precision autocast mode (master weights remain fp32)",
    )
    parser.add_argument(
        "--output-checkpoint", "--output_checkpoint",
        type=str,
        dest="output_checkpoint",
        default="nanogentzen_checkpoint.pt",
        help="Output PyTorch .pt checkpoint path",
    )
    parser.add_argument(
        "--output-safetensors", "--output_safetensors",
        type=str,
        dest="output_safetensors",
        default="model.safetensors",
        help="Output Hugging Face .safetensors path",
    )
    parser.add_argument(
        "--output-prefix", "--output_prefix",
        type=str,
        dest="output_prefix",
        default="",
        help="Optional prefix for checkpoints (e.g. nanogentzen_jev_v3)",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device_and_dtype(device_arg: str, mp_arg: str) -> Tuple[str, torch.dtype]:
    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg

    if mp_arg == "auto":
        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
    elif mp_arg == "bf16":
        dtype = torch.bfloat16
    elif mp_arg == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    return device, dtype


def build_optimizer(
    model: GentzenPolicyValueModel, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    """
    Constructs AdamW with decoupled weight decay:
      - Applies weight_decay only to 2D+ linear/attention weight matrices.
      - Excludes 1D biases, LayerNorm weights, token/positional embeddings, and pointer biases.
    """
    no_decay_names = {"pos_emb", "cls_token", "pointer_pos_bias", "pivot_bias", "tok_emb.weight"}
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim < 2
            or name in no_decay_names
            or name.endswith(".bias")
            or "ln_" in name
            or "tok_emb" in name
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr)


def split_dataset_by_trajectory(
    dataset: GentzenDataset, val_split: float, seed: int
) -> Tuple[torch.utils.data.Dataset, Optional[torch.utils.data.Dataset]]:
    """
    Splits dataset into train and validation subsets grouped by trajectory_id
    so proof steps from the same proof tree never leak between train and val.
    """
    n_total = len(dataset)
    if val_split <= 0.0 or n_total < 20:
        return dataset, None

    rng = random.Random(seed)
    traj_to_indices: Dict[int, List[int]] = {}
    for idx, item in enumerate(dataset.samples):
        if "trajectory_id" in item:
            t_id = int(item["trajectory_id"].item())
        else:
            t_id = idx
        traj_to_indices.setdefault(t_id, []).append(idx)

    traj_ids = list(traj_to_indices.keys())
    rng.shuffle(traj_ids)

    target_val = max(1, int(round(n_total * val_split)))
    val_indices: List[int] = []
    train_indices: List[int] = []

    for t_id in traj_ids:
        idxs = traj_to_indices[t_id]
        if len(val_indices) < target_val:
            val_indices.extend(idxs)
        else:
            train_indices.extend(idxs)

    if not train_indices or not val_indices:
        return dataset, None

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


@torch.no_grad()
def evaluate_loader(
    model: GentzenPolicyValueModel,
    dataloader: DataLoader,
    device: str,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    rule_correct, rule_total = 0, 0
    pivot_correct, pivot_total = 0, 0
    epistemic_correct, val_total = 0, 0
    val_abs_err = 0.0

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    for batch in dataloader:
        x = batch["input_ids"].to(device, non_blocking=True)
        y_rule = batch["target_rule"].to(device, non_blocking=True)
        y_pivot = batch["target_pivot"].to(device, non_blocking=True)

        if "target_epistemic" in batch:
            y_ep = batch["target_epistemic"].to(device, dtype=torch.long, non_blocking=True)
        else:
            y_ep = (batch["target_value"] * 2.0).round().to(device, dtype=torch.long, non_blocking=True).clamp(0, 2)
        y_val_float = batch.get("target_value", (y_ep.float() * 0.5)).to(device, non_blocking=True)

        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
            rule_logits, pivot_logits, val_pred, loss = model(
                input_ids=x,
                targets_rule=y_rule,
                targets_pivot=y_pivot,
                targets_epistemic=y_ep,
            )

        total_loss += float(loss.item())

        valid_rule = y_rule != -100
        if valid_rule.any():
            r_preds = rule_logits[valid_rule].argmax(dim=-1)
            rule_correct += int((r_preds == y_rule[valid_rule]).sum().item())
            rule_total += int(valid_rule.sum().item())

        valid_pivot = (y_pivot != -100) & (y_pivot < pivot_logits.size(-1))
        if valid_pivot.any():
            p_preds = pivot_logits[valid_pivot].argmax(dim=-1)
            pivot_correct += int((p_preds == y_pivot[valid_pivot]).sum().item())
            pivot_total += int(valid_pivot.sum().item())

        if getattr(model, "_last_epistemic_probs", None) is not None:
            ep_preds = model._last_epistemic_probs.argmax(dim=-1)
            epistemic_correct += int((ep_preds == y_ep).sum().item())

        val_abs_err += float((val_pred.detach().float() - y_val_float.float()).abs().sum().item())
        val_total += int(y_ep.numel())

    model.train()
    return {
        "loss": total_loss / max(1, len(dataloader)),
        "rule_acc": (rule_correct / rule_total * 100.0) if rule_total else 0.0,
        "pivot_acc": (pivot_correct / pivot_total * 100.0) if pivot_total else 0.0,
        "kripke_acc": (epistemic_correct / val_total * 100.0) if val_total else 0.0,
        "val_mae": (val_abs_err / val_total) if val_total else 0.0,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.output_prefix:
        if args.output_checkpoint == "nanogentzen_checkpoint.pt":
            args.output_checkpoint = f"{args.output_prefix}_checkpoint.pt"
        if args.output_safetensors == "model.safetensors":
            args.output_safetensors = f"{args.output_prefix}.safetensors"

    device, amp_dtype = resolve_device_and_dtype(args.device, args.mixed_precision)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    use_amp = (device_type == "cuda" and amp_dtype in (torch.bfloat16, torch.float16))
    print(f"=== Training nanoGentzen-v3 (Saul Kripke 3-Class) on {device} (master=fp32, amp={amp_dtype}) ===")

    tokenizer = LogicTokenizer()
    if os.path.exists(args.dataset_path):
        print(f"[*] Loading existing dataset from {args.dataset_path}...")
        dataset = GentzenDataset.load(args.dataset_path, tokenizer=tokenizer)
    else:
        print(f"[!] Dataset not found at {args.dataset_path}. Synthesizing {args.num_samples} samples on demand...")
        dataset = GentzenDataset(tokenizer, num_samples=args.num_samples, max_len=256, seed=args.seed)
        if args.dataset_path:
            os.makedirs(os.path.dirname(args.dataset_path) or ".", exist_ok=True)
            dataset.save(args.dataset_path)

    train_subset, val_subset = split_dataset_by_trajectory(dataset, args.val_split, args.seed)
    dataloader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device_type == "cuda"),
    )
    val_dataloader = (
        DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=(device_type == "cuda"),
        )
        if val_subset is not None
        else None
    )

    if os.path.exists(args.config_path):
        with open(args.config_path, "r") as f:
            cfg_dict = json.load(f)
        config = GentzenConfig(**cfg_dict)
    else:
        config = GentzenConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=256,
            n_layer=6,
            n_head=8,
            n_embd=256,
            num_rules=11,
            max_antecedents=32,
            use_pointer_pivot=True,
            pointer_dim=64,
            num_epistemic_classes=3,
        )

    # Keep master weights in float32 so small AdamW updates on LayerNorm/biases never underflow
    model = GentzenPolicyValueModel(config).to(device=device, dtype=torch.float32)

    if args.resume_safetensors and os.path.exists(args.resume_safetensors):
        if load_file is not None:
            print(f"[*] Loading base weights from {args.resume_safetensors}...")
            state_dict = load_file(args.resume_safetensors, device=device)
            if hasattr(model, "load_v2_compatible_state_dict"):
                model.load_v2_compatible_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict, strict=False)
            print("[+] Base weights loaded successfully.")
        else:
            print("[!] safetensors not installed; skipping resume_safetensors.")

    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(dataloader))
    warmup_steps = max(1, int(round(total_steps * args.warmup_ratio)))

    if total_steps > warmup_steps:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=args.lr * 0.02
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    scaler = torch.amp.GradScaler(device_type, enabled=(use_amp and amp_dtype == torch.float16))

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[*] Model parameters: {param_count:,}")
    if val_dataloader is not None:
        print(
            f"[*] Split: {len(train_subset):,} train samples | {len(val_subset):,} val samples (trajectory-grouped)"
        )
    print(f"[*] Training schedule: {args.epochs} epochs, {len(dataloader)} batches/epoch (Batch Size: {args.batch_size})")

    model.train()
    start_time = time.time()
    num_batches = len(dataloader)
    log_interval = max(10, min(200, max(1, num_batches // 5)))

    best_score = -float("inf")
    best_epoch = 0
    best_state_dict = None

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        total_loss = 0.0
        rule_correct, rule_total = 0, 0
        pivot_correct, pivot_total = 0, 0
        epistemic_correct, val_total = 0, 0
        val_abs_err = 0.0

        for step, batch in enumerate(dataloader, 1):
            x = batch["input_ids"].to(device, non_blocking=True)
            y_rule = batch["target_rule"].to(device, non_blocking=True)
            y_pivot = batch["target_pivot"].to(device, non_blocking=True)

            if "target_epistemic" in batch:
                y_ep = batch["target_epistemic"].to(device, dtype=torch.long, non_blocking=True)
            else:
                y_ep = (batch["target_value"] * 2.0).round().to(device, dtype=torch.long, non_blocking=True).clamp(0, 2)
            y_val_float = batch.get("target_value", (y_ep.float() * 0.5)).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                rule_logits, pivot_logits, val_pred, loss = model(
                    input_ids=x,
                    targets_rule=y_rule,
                    targets_pivot=y_pivot,
                    targets_epistemic=y_ep,
                )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            scheduler.step()
            total_loss += float(loss.item())

            # Rule policy accuracy on provable steps
            valid_rule = y_rule != -100
            if valid_rule.any():
                r_preds = rule_logits[valid_rule].argmax(dim=-1)
                rule_correct += int((r_preds == y_rule[valid_rule]).sum().item())
                rule_total += int(valid_rule.sum().item())

            # Span pointer pivot accuracy on left-rule steps
            valid_pivot = (y_pivot != -100) & (y_pivot < pivot_logits.size(-1))
            if valid_pivot.any():
                p_preds = pivot_logits[valid_pivot].argmax(dim=-1)
                pivot_correct += int((p_preds == y_pivot[valid_pivot]).sum().item())
                pivot_total += int(valid_pivot.sum().item())

            # Saul Kripke 3-Class Epistemic Accuracy
            if getattr(model, "_last_epistemic_probs", None) is not None:
                ep_preds = model._last_epistemic_probs.argmax(dim=-1)
                epistemic_correct += int((ep_preds == y_ep).sum().item())

            val_abs_err += float((val_pred.detach().float() - y_val_float.float()).abs().sum().item())
            val_total += int(y_ep.numel())

            if step % log_interval == 0 or step == num_batches:
                curr_ep_acc = (epistemic_correct / val_total * 100.0) if val_total else 0.0
                curr_r_acc = (rule_correct / rule_total * 100.0) if rule_total else 0.0
                curr_p_acc = (pivot_correct / pivot_total * 100.0) if pivot_total else 0.0
                elapsed_s = time.time() - epoch_start
                print(
                    f"  [Epoch {epoch:02d} | Step {step:4d}/{num_batches:4d}] "
                    f"Loss: {loss.item():.4f} (Avg: {total_loss/step:.4f}) | "
                    f"Kripke Acc: {curr_ep_acc:5.2f}% | Rule Acc: {curr_r_acc:5.2f}% | "
                    f"Pivot Acc: {curr_p_acc:5.2f}% | Elapsed: {elapsed_s:.1f}s"
                )

        avg_loss = total_loss / max(1, len(dataloader))
        r_acc = (rule_correct / rule_total * 100.0) if rule_total else 0.0
        p_acc = (pivot_correct / pivot_total * 100.0) if pivot_total else 0.0
        e_acc = (epistemic_correct / val_total * 100.0) if val_total else 0.0
        v_mae = (val_abs_err / val_total) if val_total else 0.0

        print(
            f"--> Epoch {epoch:02d}/{args.epochs:02d} Train   | "
            f"Loss: {avg_loss:.4f} | Rule Acc: {r_acc:5.2f}% | "
            f"Span Pointer Acc: {p_acc:5.2f}% | 3-Class Kripke Acc: {e_acc:5.2f}% | "
            f"Val MAE: {v_mae:.4f} | Time: {time.time() - epoch_start:.2f}s"
        )

        if val_dataloader is not None:
            val_metrics = evaluate_loader(model, val_dataloader, device, amp_dtype, use_amp)
            epoch_score = (
                0.40 * val_metrics["kripke_acc"]
                + 0.30 * val_metrics["rule_acc"]
                + 0.30 * val_metrics["pivot_acc"]
                - 5.0 * val_metrics["loss"]
            )
            is_best = epoch_score > best_score
            if is_best:
                best_score = epoch_score
                best_epoch = epoch
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = " [BEST]" if is_best else ""
            print(
                f"--> Epoch {epoch:02d}/{args.epochs:02d} Val     | "
                f"Loss: {val_metrics['loss']:.4f} | Rule Acc: {val_metrics['rule_acc']:5.2f}% | "
                f"Span Pointer Acc: {val_metrics['pivot_acc']:5.2f}% | 3-Class Kripke Acc: {val_metrics['kripke_acc']:5.2f}% | "
                f"Val MAE: {val_metrics['val_mae']:.4f}{marker}"
            )
        else:
            epoch_score = 0.40 * e_acc + 0.30 * r_acc + 0.30 * p_acc - 5.0 * avg_loss
            if epoch_score >= best_score:
                best_score = epoch_score
                best_epoch = epoch
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"[*] Restored best model weights from Epoch {best_epoch:02d}")

    # Save output artifacts
    out_dir = os.path.dirname(args.output_checkpoint)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.save(
        {"model_state": model.state_dict(), "config": config, "best_epoch": best_epoch},
        args.output_checkpoint,
    )
    print(f"[+] PyTorch checkpoint saved to {args.output_checkpoint}")

    if args.output_safetensors and save_file is not None:
        sft_dir = os.path.dirname(args.output_safetensors)
        if sft_dir:
            os.makedirs(sft_dir, exist_ok=True)
        fp32_state = {k: v.detach().cpu().float().contiguous() for k, v in model.state_dict().items()}
        cfg_dict = asdict(config) if hasattr(config, "__dataclass_fields__") else (vars(config) if hasattr(config, "__dict__") else {})
        metadata = {str(k): str(v) for k, v in cfg_dict.items() if not k.startswith("_")}
        save_file(fp32_state, args.output_safetensors, metadata=metadata)
        print(f"[+] Safetensors model saved to {args.output_safetensors}")

    total_time = time.time() - start_time
    print(f"\n[OK] Training completed in {total_time:.2f}s ({total_time/60:.2f} min).")


if __name__ == "__main__":
    main()
