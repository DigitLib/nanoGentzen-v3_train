"""
generate_dataset.py (nanoGentzen-v3 Kripke + Jev)
Synthesizes, exports, and previews 1:1:1 balanced 3-Class Saul Kripke datasets:
  - Class 2 (PROVEN_TRUE, target_epistemic=2, value=1.0): Certified constructive proofs
  - Class 1 (INTUITIONISTIC_UNDECIDED, target_epistemic=1, value=0.5): Kripke open countermodels & fallacies
  - Class 0 (PROVEN_FALSE, target_epistemic=0, value=0.0): Certified constructive refutations & contradictions
"""
import argparse
from collections import Counter
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch

try:
    from nanogentzen.dataset import (
        VARS_POOL,
        exhaustive_solver,
        generate_random_formula,
        synthesize_constructive_theorem_trace,
        synthesize_horn_policy_trace,
        synthesize_refuted_sequent,
    )
    from nanogentzen.kernel import And, Imp, Not, Or, RULES, Sequent, Var
    from nanogentzen.tokenization_nanogentzen import LogicTokenizer
except ImportError:
    from dataset import (
        VARS_POOL,
        exhaustive_solver,
        generate_random_formula,
        synthesize_constructive_theorem_trace,
        synthesize_horn_policy_trace,
        synthesize_refuted_sequent,
    )
    from kernel import And, Imp, Not, Or, RULES, Sequent, Var
    from tokenization_nanogentzen import LogicTokenizer


def generate_rich_dataset(
    tokenizer: LogicTokenizer,
    num_samples: int = 12000,
    max_len: int = 256,
    seed: Optional[int] = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, torch.Tensor]]]:
    """
    Synthesizes certified 3-Class Kripke transitions:
      1/3 Class 2 (True)
      1/3 Class 1 (Undecided)
      1/3 Class 0 (False)
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    rich_records: List[Dict[str, Any]] = []
    tensor_samples: List[Dict[str, torch.Tensor]] = []

    target_pos = num_samples // 3
    target_undec = num_samples // 3
    target_false = num_samples - target_pos - target_undec

    count_pos = 0
    count_undec = 0
    count_false = 0

    def add_record(
        seq: Sequent,
        rule_name: str,
        rule_idx: int,
        pivot_idx: int,
        epistemic_cls: int,
        target_val: float,
        root_seq: Sequent,
        step_idx: int,
        total_steps: int,
        curriculum: str,
        traj_id: int,
    ):
        seq_str = seq.to_str()
        enc = tokenizer.encode(seq_str)[:max_len]
        padded = enc + [tokenizer.pad_id] * (max_len - len(enc))
        is_left_rule = rule_name.startswith("L_")
        pivot_clamped = min(max(pivot_idx, 0), 31) if (is_left_rule and pivot_idx >= 0) else -100

        tensor_samples.append(
            {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "target_rule": torch.tensor(rule_idx, dtype=torch.long),
                "target_pivot": torch.tensor(pivot_clamped, dtype=torch.long),
                "target_epistemic": torch.tensor(epistemic_cls, dtype=torch.long),
                "target_value": torch.tensor(target_val, dtype=torch.float32),
                "trajectory_id": torch.tensor(traj_id, dtype=torch.long),
            }
        )
        rich_records.append(
            {
                "sample_id": len(tensor_samples),
                "trajectory_id": traj_id,
                "curriculum": curriculum,
                "sequent": seq_str,
                "rule": rule_name,
                "rule_idx": rule_idx,
                "pivot": pivot_clamped,
                "target_epistemic": epistemic_cls,
                "target_value": target_val,
                "root_sequent": root_seq.to_str(),
                "trace_step": step_idx,
                "total_trace_steps": total_steps,
                "input_ids": enc,
                "token_length": len(enc),
            }
        )

    print(f"[*] Synthesizing {num_samples} 3-Class Kripke transitions (seed={seed})...")
    print(f"    Target Split: {target_pos} True | {target_undec} Undecided | {target_false} False")

    start_t = time.time()
    trajectory_count = 0

    while len(tensor_samples) < num_samples:
        trajectory_count += 1

        # -------------------------------------------------------------
        # 1. CLASS 2: PROVEN_TRUE
        # -------------------------------------------------------------
        if count_pos < target_pos:
            mode = random.random()
            if mode < 0.55:
                # Multi-hop Horn policy trace with Span Pointer pivots (0..31)
                provable_root, trace, _, _ = synthesize_horn_policy_trace()
                total_steps = len(trace)
                for step_idx, (s, rule, pivot) in enumerate(trace):
                    if count_pos >= target_pos or len(tensor_samples) >= num_samples:
                        break
                    r_idx = RULES.index(rule) if rule in RULES else -100
                    add_record(
                        seq=s,
                        rule_name=rule,
                        rule_idx=r_idx,
                        pivot_idx=pivot,
                        epistemic_cls=2,
                        target_val=1.0,
                        root_seq=provable_root,
                        step_idx=step_idx,
                        total_steps=total_steps,
                        curriculum="horn_policy_provable",
                        traj_id=trajectory_count,
                    )
                    count_pos += 1
            elif mode < 0.80:
                # Curated constructive & refutation theorems across all 11 rules
                root_seq, trace, kind = synthesize_constructive_theorem_trace()
                total_steps = len(trace)
                for step_idx, (s, rule, pivot) in enumerate(trace):
                    if count_pos >= target_pos or len(tensor_samples) >= num_samples:
                        break
                    r_idx = RULES.index(rule) if rule in RULES else -100
                    add_record(
                        seq=s,
                        rule_name=rule,
                        rule_idx=r_idx,
                        pivot_idx=pivot,
                        epistemic_cls=2,
                        target_val=1.0,
                        root_seq=root_seq,
                        step_idx=step_idx,
                        total_steps=total_steps,
                        curriculum=f"constructive_{kind}",
                        traj_id=trajectory_count,
                    )
                    count_pos += 1
            else:
                # Random constructive theorem
                ant_count = random.randint(0, 3)
                vars_sub = random.sample(VARS_POOL, k=4)
                gamma = tuple(
                    generate_random_formula(depth=random.randint(0, 2), vars_subset=vars_sub)
                    for _ in range(ant_count)
                )
                delta = (generate_random_formula(depth=random.randint(1, 2), vars_subset=vars_sub),)
                seq = Sequent(gamma, delta)
                trace = exhaustive_solver(seq, max_depth=6)
                if trace is not None:
                    total_steps = len(trace)
                    for step_idx, (s, rule, pivot) in enumerate(trace):
                        if count_pos >= target_pos or len(tensor_samples) >= num_samples:
                            break
                        r_idx = RULES.index(rule) if rule in RULES else -100
                        add_record(
                            seq=s,
                            rule_name=rule,
                            rule_idx=r_idx,
                            pivot_idx=pivot,
                            epistemic_cls=2,
                            target_val=1.0,
                            root_seq=seq,
                            step_idx=step_idx,
                            total_steps=total_steps,
                            curriculum="random_theorem_provable",
                            traj_id=trajectory_count,
                        )
                        count_pos += 1

        # -------------------------------------------------------------
        # 2. CLASS 1: INTUITIONISTIC_UNDECIDED
        # -------------------------------------------------------------
        if count_undec < target_undec and len(tensor_samples) < num_samples:
            trajectory_count += 1
            mode = random.random()
            if mode < 0.50:
                # Enterprise policy missing required premise
                _, _, undecided_seq, _ = synthesize_horn_policy_trace()
                add_record(
                    seq=undecided_seq,
                    rule_name="UNDECIDED",
                    rule_idx=-100,
                    pivot_idx=-100,
                    epistemic_cls=1,
                    target_val=0.5,
                    root_seq=undecided_seq,
                    step_idx=0,
                    total_steps=1,
                    curriculum="policy_missing_premise",
                    traj_id=trajectory_count,
                )
                count_undec += 1
            elif mode < 0.75:
                # Known intuitionistic non-theorems & classic fallacies
                p, q = random.sample(VARS_POOL, 2)
                fallacies = [
                    (Sequent((), (Or(p, Not(p)),)), "law_of_excluded_middle"),
                    (Sequent((), (Imp(Imp(Imp(p, q), p), p),)), "peirce_law"),
                    (Sequent((Not(Not(p)),), (p,)), "double_negation_elimination"),
                    (Sequent((Imp(p, q), q), (p,)), "affirming_consequent"),
                    (Sequent((Imp(p, q), Not(p)), (Not(q),)), "denying_antecedent"),
                    (Sequent((p,), (q,)), "unrelated_premise"),
                ]
                f_seq, f_name = random.choice(fallacies)
                add_record(
                    seq=f_seq,
                    rule_name="UNDECIDED",
                    rule_idx=-100,
                    pivot_idx=-100,
                    epistemic_cls=1,
                    target_val=0.5,
                    root_seq=f_seq,
                    step_idx=0,
                    total_steps=1,
                    curriculum=f_name,
                    traj_id=trajectory_count,
                )
                count_undec += 1
            else:
                # Random open countermodel
                vars_sub = random.sample(VARS_POOL, k=3)
                gamma = (generate_random_formula(depth=1, vars_subset=vars_sub),)
                delta = (generate_random_formula(depth=random.randint(1, 2), vars_subset=vars_sub),)
                seq = Sequent(gamma, delta)
                if (
                    exhaustive_solver(seq, max_depth=5) is None
                    and exhaustive_solver(Sequent(gamma, (Not(delta[0]),)), max_depth=5) is None
                ):
                    add_record(
                        seq=seq,
                        rule_name="UNDECIDED",
                        rule_idx=-100,
                        pivot_idx=-100,
                        epistemic_cls=1,
                        target_val=0.5,
                        root_seq=seq,
                        step_idx=0,
                        total_steps=1,
                        curriculum="open_kripke_countermodel",
                        traj_id=trajectory_count,
                    )
                    count_undec += 1

        # -------------------------------------------------------------
        # 3. CLASS 0: PROVEN_FALSE
        # -------------------------------------------------------------
        if count_false < target_false and len(tensor_samples) < num_samples:
            trajectory_count += 1
            mode = random.random()
            if mode < 0.50:
                # Policy with shuffled contradiction/refutation injected
                _, _, _, refuted_seq = synthesize_horn_policy_trace()
                add_record(
                    seq=refuted_seq,
                    rule_name="REFUTED",
                    rule_idx=-100,
                    pivot_idx=-100,
                    epistemic_cls=0,
                    target_val=0.0,
                    root_seq=refuted_seq,
                    step_idx=0,
                    total_steps=1,
                    curriculum="policy_refutation_contradiction",
                    traj_id=trajectory_count,
                )
                count_false += 1
            else:
                # Synthesized constructive refutation pattern with shuffled antecedents
                refuted_seq = synthesize_refuted_sequent()
                add_record(
                    seq=refuted_seq,
                    rule_name="REFUTED",
                    rule_idx=-100,
                    pivot_idx=-100,
                    epistemic_cls=0,
                    target_val=0.0,
                    root_seq=refuted_seq,
                    step_idx=0,
                    total_steps=1,
                    curriculum="constructive_refutation",
                    traj_id=trajectory_count,
                )
                count_false += 1

    elapsed = time.time() - start_t
    print(
        f"[+] Generated {len(tensor_samples)} transitions from {trajectory_count} proof trajectories in {elapsed:.2f}s"
    )
    return rich_records, tensor_samples


def print_dataset_summary(records: List[Dict[str, Any]], num_preview: int = 5):
    """Prints distribution statistics across rules, pivots, and 3-class epistemic states."""
    rule_counts = Counter(r["rule"] for r in records)
    ep_counts = Counter(r["target_epistemic"] for r in records)
    pivot_counts = Counter(r["pivot"] for r in records if r["pivot"] >= 0)
    total = len(records)

    print("\n" + "=" * 70)
    print(f" SAUL KRIPKE 3-CLASS DATASET SUMMARY (Total Samples: {total})")
    print("=" * 70)
    print(f"{'Rule / State':<16} | {'Count':<8} | {'Percentage':<10}")
    print("-" * 70)
    for rule in RULES + ["UNDECIDED", "REFUTED"]:
        cnt = rule_counts.get(rule, 0)
        pct = (cnt / total * 100) if total > 0 else 0
        if cnt > 0:
            print(f"{rule:<16} | {cnt:<8} | {pct:>6.2f}%")
    print("-" * 70)

    print("\nEpistemic 3-Class Target Distribution:")
    ep_labels = {
        2: "Class 2: PROVEN_TRUE (1.0)",
        1: "Class 1: INTUITIONISTIC_UNDECIDED (0.5)",
        0: "Class 0: PROVEN_FALSE / REFUTED (0.0)",
    }
    for cls_idx in [2, 1, 0]:
        cnt = ep_counts.get(cls_idx, 0)
        pct = (cnt / total * 100) if total > 0 else 0
        print(f"  {ep_labels[cls_idx]:<42}: {cnt:<6} ({pct:>5.2f}%)")

    print("\nAntecedent Span Pointer Pivot Distribution (0..31, Left Rules Only):")
    piv_str = ", ".join(f"p{p}:{pivot_counts.get(p, 0)}" for p in range(16))
    print(f"  Pivots 0..15 : {piv_str}")
    piv_str_hi = ", ".join(f"p{p}:{pivot_counts.get(p, 0)}" for p in range(16, 32))
    print(f"  Pivots 16..31: {piv_str_hi}")

    avg_tok_len = sum(r["token_length"] for r in records) / total if total else 0
    max_tok_len = max((r["token_length"] for r in records), default=0)
    print(f"\nSequent Token Length: Avg = {avg_tok_len:.1f} tokens | Max = {max_tok_len} tokens")

    print("\n" + "=" * 70)
    print(f" DATASET PREVIEW (First {min(num_preview, total)} samples)")
    print("=" * 70)
    for r in records[:num_preview]:
        print(f"Sample #{r['sample_id']} [{r['curriculum']}]:")
        print(f"  Sequent         : {r['sequent']}")
        print(f"  Target Rule     : {r['rule']} (ID: {r['rule_idx']})")
        print(f"  Target Pivot    : {r['pivot']}")
        print(f"  Epistemic Class : {r['target_epistemic']} (Value: {r['target_value']})")
        print("-" * 70)


def save_dataset(
    rich_records: List[Dict[str, Any]],
    tensor_samples: List[Dict[str, torch.Tensor]],
    output_dir: str = "data",
    jsonl_filename: str = "gentzen_dataset.jsonl",
    pt_filename: str = "gentzen_dataset.pt",
    export_jsonl: bool = True,
    export_pt: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    if export_jsonl:
        jsonl_path = os.path.join(output_dir, jsonl_filename)
        print(f"[*] Saving human-readable dataset to {jsonl_path}...")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in rich_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[+] JSONL saved ({os.path.getsize(jsonl_path) / (1024 * 1024):.2f} MB)")

    if export_pt:
        pt_path = os.path.join(output_dir, pt_filename)
        print(f"[*] Saving PyTorch tensor dataset to {pt_path}...")
        torch.save(tensor_samples, pt_path)
        print(f"[+] PyTorch dataset saved ({os.path.getsize(pt_path) / (1024 * 1024):.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3-Class Saul Kripke nanoGentzen-v3 dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-samples", "--num_samples", type=int, dest="num_samples", default=12000, help="Number of transitions to generate")
    parser.add_argument("--max-len", "--max_len", type=int, dest="max_len", default=256, help="Maximum token length per sequent")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", "--output_dir", type=str, dest="output_dir", default="data", help="Output directory")
    parser.add_argument("--no-jsonl", action="store_true", help="Skip exporting .jsonl")
    parser.add_argument("--no-pt", action="store_true", help="Skip exporting .pt")
    parser.add_argument("--preview", type=int, default=5, help="Number of samples to preview")
    args = parser.parse_args()

    tokenizer = LogicTokenizer()
    rich_records, tensor_samples = generate_rich_dataset(
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        max_len=args.max_len,
        seed=args.seed,
    )
    print_dataset_summary(rich_records, num_preview=args.preview)
    save_dataset(
        rich_records,
        tensor_samples,
        output_dir=args.output_dir,
        export_jsonl=not args.no_jsonl,
        export_pt=not args.no_pt,
    )
    print("\n[OK] Saul Kripke 3-Class dataset generation and export complete!")


if __name__ == "__main__":
    main()
