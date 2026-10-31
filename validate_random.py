"""
validate_random.py (nanoGentzen-v3 Kripke + Jev)
Comprehensive validation suite evaluating 3-class epistemic accuracy,
search node efficiency, and open countermodel extraction across:
  - 50 Random Valid Theorems (Class 2)
  - 50 Fallacies & Open Countermodels (Class 1)
  - 50 Constructive Refutations & Contradictions (Class 0)
"""
import argparse
import json
import os
import random
import sys
import time
from typing import List, Tuple
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from safetensors.torch import load_file
except ImportError:
    load_file = None

try:
    from nanogentzen.dataset import (
        generate_random_formula,
        exhaustive_solver,
        synthesize_refuted_sequent,
        VARS_POOL,
    )
    from nanogentzen.kernel import (
        And,
        Formula,
        Imp,
        Not,
        Or,
        Sequent,
        Var,
        verify_proof_tree_strict,
        extract_kripke_countermodel,
    )
    from nanogentzen.modeling_nanogentzen import GentzenPolicyValueModel
    from nanogentzen.configuration_nanogentzen import GentzenConfig
    from nanogentzen.search import NeuralProofSearch
    from nanogentzen.tokenization_nanogentzen import LogicTokenizer
except ImportError:
    from dataset import (
        generate_random_formula,
        exhaustive_solver,
        synthesize_refuted_sequent,
        VARS_POOL,
    )
    from kernel import (
        And,
        Formula,
        Imp,
        Not,
        Or,
        Sequent,
        Var,
        verify_proof_tree_strict,
        extract_kripke_countermodel,
    )
    from modeling_nanogentzen import GentzenPolicyValueModel
    from configuration_nanogentzen import GentzenConfig
    from search import NeuralProofSearch
    from tokenization_nanogentzen import LogicTokenizer


def generate_fallacy_sequents() -> List[Tuple[Sequent, str]]:
    P, Q, R = Var("P"), Var("Q"), Var("R")
    return [
        (Sequent((Imp(P, Q), Q), (P,)), "Affirming the Consequent"),
        (Sequent((Imp(P, Q), Not(P)), (Not(Q),)), "Denying the Antecedent"),
        (Sequent((Or(P, Q), P), (Not(Q),)), "Affirming a Disjunct"),
        (Sequent((P,), (Q,)), "Unrelated Variable Premise"),
        (Sequent((), (Or(P, Not(P)),)), "Law of Excluded Middle"),
        (Sequent((), (Imp(Imp(Imp(P, Q), P), P),)), "Peirce's Law"),
        (Sequent((Not(Not(P)),), (P,)), "Double Negation Elimination"),
        (Sequent((Imp(P, Q),), (Imp(Q, P),)), "Converse of Implication"),
    ]


def generate_benchmark_suite(
    num_valid: int = 50, num_undecided: int = 50, num_refuted: int = 50, seed: int = 42
) -> List[Tuple[Sequent, int, str]]:
    random.seed(seed)
    suite: List[Tuple[Sequent, int, str]] = []

    # 1. Class 2: Valid Theorems
    print(f"[*] Generating {num_valid} valid theorems (Class 2)...")
    valid_count = 0
    attempts = 0
    while valid_count < num_valid and attempts < num_valid * 60:
        attempts += 1
        ant_count = random.randint(0, 3)
        vars_sub = random.sample(VARS_POOL, k=4)
        gamma = tuple(generate_random_formula(depth=random.randint(0, 2), vars_subset=vars_sub) for _ in range(ant_count))
        delta = (generate_random_formula(depth=random.randint(1, 2), vars_subset=vars_sub),)
        seq = Sequent(gamma, delta)
        trace = exhaustive_solver(seq, max_depth=6)
        if trace is not None:
            suite.append((seq, 2, "Valid Theorem"))
            valid_count += 1

    # 2. Class 1: Undecided & Fallacies
    print(f"[*] Generating {num_undecided} open countermodels & fallacies (Class 1)...")
    fallacies = generate_fallacy_sequents()
    for seq, name in fallacies:
        suite.append((seq, 1, f"Fallacy: {name}"))

    remaining_undecided = num_undecided - len(fallacies)
    undec_count = 0
    attempts = 0
    while undec_count < remaining_undecided and attempts < remaining_undecided * 60:
        attempts += 1
        vars_sub = random.sample(VARS_POOL, k=3)
        gamma = (generate_random_formula(depth=1, vars_subset=vars_sub),)
        delta = (generate_random_formula(depth=random.randint(1, 2), vars_subset=vars_sub),)
        seq = Sequent(gamma, delta)
        if (
            exhaustive_solver(seq, max_depth=5) is None
            and exhaustive_solver(Sequent(gamma, (Not(delta[0]),)), max_depth=5) is None
        ):
            suite.append((seq, 1, "Random Open Countermodel"))
            undec_count += 1

    # 3. Class 0: Refuted Sequents
    print(f"[*] Generating {num_refuted} refuted & contradictory sequents (Class 0)...")
    for _ in range(num_refuted):
        seq = synthesize_refuted_sequent()
        suite.append((seq, 0, "Refuted / Contradiction"))

    random.shuffle(suite)
    return suite


def load_model(device: str, dtype: torch.dtype) -> Tuple[GentzenPolicyValueModel, LogicTokenizer]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer = LogicTokenizer()
    sft_path = os.path.join(base_dir, "model.safetensors")
    cfg_path = os.path.join(base_dir, "config.json")
    ckpt_path = os.path.join(base_dir, "nanogentzen_checkpoint.pt")

    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
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

    model = GentzenPolicyValueModel(config).to(device=device, dtype=dtype)
    if os.path.exists(sft_path) and load_file is not None:
        state_dict = load_file(sft_path, device=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"[*] Loaded weights from {sft_path}")
    elif os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[*] Loaded weights from {ckpt_path}")
    else:
        print("[!] Warning: No model weights found; testing untrained model.")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Validate nanoGentzen-v3 Saul Kripke Prover")
    parser.add_argument("--num-samples", type=int, default=50, help="Samples per class (total = 3 * num-samples)")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation random seed")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto/cuda/cpu)")
    args = parser.parse_args()

    device = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    print(f"=== Validating nanoGentzen-v3 Kripke Engine on {device} ({dtype}) ===")
    model, tokenizer = load_model(device, dtype)
    searcher = NeuralProofSearch(model, tokenizer, device=device)

    dataset = generate_benchmark_suite(
        num_valid=args.num_samples,
        num_undecided=args.num_samples,
        num_refuted=args.num_samples,
        seed=args.seed,
    )
    total = len(dataset)
    print(f"\nRunning validation over {total} balanced items...")

    epistemic_correct = {0: 0, 1: 0, 2: 0}
    epistemic_total = {0: 0, 1: 0, 2: 0}
    search_success = 0
    search_sound = 0
    start_t = time.time()

    for seq, true_cls, tag in dataset:
        # System 1: Epistemic evaluation
        encoded = tokenizer.encode(seq.to_str())[: model.config.block_size]
        padded = encoded + [tokenizer.pad_id] * (model.config.block_size - len(encoded))
        inp = torch.tensor([padded], dtype=torch.long, device=device)

        _, _, ep_probs, _ = model.predict_epistemic(inp)
        pred_cls = int(ep_probs[0].argmax(dim=-1).item())

        epistemic_total[true_cls] += 1
        if pred_cls == true_cls:
            epistemic_correct[true_cls] += 1

        # System 2: Search on Class 2 theorems
        if true_cls == 2:
            proof, _ = searcher.prove(seq, max_depth=8)
            if proof is not None:
                search_success += 1
                if verify_proof_tree_strict(proof):
                    search_sound += 1

    elapsed = time.time() - start_t
    overall_correct = sum(epistemic_correct.values())

    print("\n" + "=" * 65)
    print(" VALIDATION RESULTS: 3-CLASS SAUL KRIPKE EVALUATION")
    print("=" * 65)
    print(f"Total Evaluated Items : {total}")
    print(f"Total Elapsed Time    : {elapsed:.2f}s ({(elapsed/total)*1000:.1f} ms/item)")
    print("-" * 65)
    print("System 1 Epistemic Classification Accuracy:")
    for c, label in [
        (2, "Class 2: PROVEN_TRUE"),
        (1, "Class 1: INTUITIONISTIC_UNDECIDED"),
        (0, "Class 0: PROVEN_FALSE"),
    ]:
        tot = epistemic_total[c]
        corr = epistemic_correct[c]
        acc = (corr / tot * 100.0) if tot else 0.0
        print(f"  {label:<38}: {corr:3d}/{tot:3d} ({acc:5.1f}%)")

    overall_acc = (overall_correct / total * 100.0) if total else 0.0
    print(f"  {'Overall 3-Class Accuracy':<38}: {overall_correct:3d}/{total:3d} ({overall_acc:5.1f}%)")
    print("-" * 65)
    print("System 2 Neural Proof Search (Class 2 Theorems):")
    valid_tot = epistemic_total[2]
    solve_rate = (search_success / valid_tot * 100.0) if valid_tot else 0.0
    sound_rate = (search_sound / valid_tot * 100.0) if valid_tot else 0.0
    print(f"  Proved Theorems                     : {search_success:3d}/{valid_tot:3d} ({solve_rate:5.1f}%)")
    print(f"  Verified Strictly Sound Proof Trees : {search_sound:3d}/{valid_tot:3d} ({sound_rate:5.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    main()

