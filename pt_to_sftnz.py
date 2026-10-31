"""
pt_to_sftnz.py (nanoGentzen-v3 Kripke + Jev)
Converts PyTorch checkpoint (.pt) to Hugging Face safetensors format
with embedded configuration metadata.
"""
import argparse
from dataclasses import asdict
import json
import os
import sys
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from safetensors.torch import save_file
except ImportError:
    save_file = None


def convert_checkpoint(ckpt_path: str = "nanogentzen_checkpoint.pt", output_sft: str = "model.safetensors"):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    if save_file is None:
        raise ImportError("safetensors package is required. Install via 'pip install safetensors'")

    print(f"[*] Loading PyTorch checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_state = ckpt.get("model_state", ckpt)
    config = ckpt.get("config", None)

    # Convert state dict tensors to contiguous fp32
    fp32_state = {k: v.detach().cpu().float().contiguous() for k, v in model_state.items()}

    # Format metadata header
    metadata = {}
    if config is not None:
        cfg_dict = asdict(config) if hasattr(config, "__dataclass_fields__") else (vars(config) if hasattr(config, "__dict__") else {})
        metadata = {str(k): str(v) for k, v in cfg_dict.items() if not k.startswith("_")}

        # Also export clean config.json
        cfg_out = os.path.join(os.path.dirname(output_sft) or ".", "config.json")
        with open(cfg_out, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in cfg_dict.items() if not k.startswith("_")}, f, indent=2)
        print(f"[+] Exported clean architecture config to {cfg_out}")

    print(f"[*] Saving safetensors file to {output_sft}...")
    save_file(fp32_state, output_sft, metadata=metadata)
    size_mb = os.path.getsize(output_sft) / (1024 * 1024)
    print(f"[+] Successfully converted to {output_sft} ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert nanoGentzen .pt to model.safetensors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default="nanogentzen_checkpoint.pt", help="Path to .pt checkpoint")
    parser.add_argument("--output", type=str, default="model.safetensors", help="Path to output .safetensors")
    args = parser.parse_args()

    convert_checkpoint(args.checkpoint, args.output)


if __name__ == "__main__":
    main()

