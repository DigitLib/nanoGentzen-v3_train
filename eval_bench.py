"""
eval_bench.py (nanoGentzen-v3 Kripke + Jev)
Benchmark suite testing constructive theorem proving, 3-class Saul Kripke
epistemic valuation, strict proof-tree soundness, and open countermodel extraction.
"""
import json
import os
import sys
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from nanogentzen.kernel import (
        And,
        Formula,
        Imp,
        Not,
        Or,
        Sequent,
        Var,
        canonicalize_sequent,
        verify_proof_tree_strict,
        extract_kripke_countermodel,
    )
    from nanogentzen.modeling_nanogentzen import GentzenPolicyValueModel
    from nanogentzen.configuration_nanogentzen import GentzenConfig
    from nanogentzen.tokenization_nanogentzen import LogicTokenizer
    from nanogentzen.search import NeuralProofSearch
except ImportError:
    from kernel import (
        And,
        Formula,
        Imp,
        Not,
        Or,
        Sequent,
        Var,
        canonicalize_sequent,
        verify_proof_tree_strict,
        extract_kripke_countermodel,
    )
    from modeling_nanogentzen import GentzenPolicyValueModel
    from configuration_nanogentzen import GentzenConfig
    from tokenization_nanogentzen import LogicTokenizer
    from search import NeuralProofSearch

try:
    from safetensors.torch import load_file
except ImportError:
    load_file = None


def load_model_and_searcher(device: str = "cuda"):
    tokenizer = LogicTokenizer()
    base_dir = os.path.dirname(os.path.abspath(__file__))
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

    model = GentzenPolicyValueModel(config).to(device)

    if os.path.exists(sft_path) and load_file is not None:
        state_dict = load_file(sft_path, device=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"[*] Loaded weights from {sft_path}")
    elif os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[*] Loaded weights from {ckpt_path}")
    else:
        print("[!] No trained weights found on disk; running benchmark with initialized model.")

    searcher = NeuralProofSearch(model, tokenizer, device=device)
    return model, tokenizer, searcher


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Running nanoGentzen-v3 Kripke Benchmark Suite on {device} ===")
    model, tokenizer, searcher = load_model_and_searcher(device)

    P, Q, R = Var("P"), Var("Q"), Var("R")
    kyc_ok = Var("kyc_verified")
    risk_ok = Var("within_risk_limit")
    dev_ok = Var("device_trusted")
    approved = Var("can_approve")

    benchmarks = [
        # --- CLASS 2: PROVEN_TRUE ---
        ("1. Identity Theorem", Sequent((), (Imp(P, P),)), 2),
        ("2. Conjunction Intro", Sequent((P, Q), (And(P, Q),)), 2),
        ("3. Implication Transitivity", Sequent((Imp(P, Q), Imp(Q, R)), (Imp(P, R),)), 2),
        ("4. Constructive De Morgan", Sequent((Not(Or(P, Q)),), (And(Not(P), Not(Q)),)), 2),
        (
            "5. Enterprise Policy Chain",
            Sequent(
                (kyc_ok, risk_ok, Imp(And(kyc_ok, risk_ok), approved)),
                (approved,),
            ),
            2,
        ),
        # --- CLASS 1: INTUITIONISTIC_UNDECIDED ---
        ("6. Law of Excluded Middle (LEM)", Sequent((), (Or(P, Not(P)),)), 1),
        ("7. Peirce's Law (LI Invalid)", Sequent((), (Imp(Imp(Imp(P, Q), P), P),)), 1),
        ("8. Affirming the Consequent", Sequent((Imp(P, Q), Q), (P,)), 1),
        (
            "9. Missing Fact Policy",
            Sequent(
                (kyc_ok, Imp(And(kyc_ok, dev_ok), approved)),
                (approved,),
            ),
            1,
        ),
        # --- CLASS 0: PROVEN_FALSE ---
        ("10. Contradiction on Goal", Sequent((Not(P),), (P,)), 0),
        ("11. Modus Tollens Refutation", Sequent((Imp(P, Q), Not(Q)), (P,)), 0),
        ("12. Inconsistent Premise Base", Sequent((P, Not(P)), (Q,)), 0),
    ]

    print("\n" + "=" * 85)
    print(f"{'Benchmark Goal':<34} | {'Target':<7} | {'System 1 P(F,U,T) / V':<22} | {'System 2 Status'}")
    print("=" * 85)

    s1_correct = 0
    s2_correct = 0

    for name, seq, expected_cls in benchmarks:
        # System 1: Epistemic single-pass forward (with alpha-renaming canonicalization)
        eval_seq = canonicalize_sequent(seq)[0]
        encoded = tokenizer.encode(eval_seq.to_str())[: model.config.block_size]
        padded = encoded + [tokenizer.pad_id] * (model.config.block_size - len(encoded))
        inp = torch.tensor([padded], dtype=torch.long, device=device)

        _, _, ep_probs, val_score = model.predict_epistemic(inp)
        p_f = float(ep_probs[0, 0].item())
        p_u = float(ep_probs[0, 1].item())
        p_t = float(ep_probs[0, 2].item())
        pred_cls = int(ep_probs[0].argmax(dim=-1).item())

        if pred_cls == expected_cls:
            s1_correct += 1

        # System 2: Neural Proof Search + Kripke Countermodel
        proof, countermodel = searcher.prove(seq, max_depth=8)

        if proof:
            sound = verify_proof_tree_strict(proof)
            cm_check = extract_kripke_countermodel(seq)
            if cm_check.get("refutation_reason", "").startswith("Internal premise contradiction"):
                status = f"REFUTED (Premise Contradiction / Ex Falso, Sound={sound})"
                if expected_cls == 0 and sound:
                    s2_correct += 1
            else:
                status = f"PROVEN (Sound={sound})"
                if expected_cls == 2 and sound:
                    s2_correct += 1
        else:
            diag = countermodel.get("diagnosis") if countermodel else None
            if diag == "PROVEN_FALSE":
                status = "REFUTED (Contradiction Certified)"
                if expected_cls == 0:
                    s2_correct += 1
            elif countermodel and countermodel.get("missing_facts"):
                missing = ",".join(countermodel["missing_facts"])
                status = f"UNDECIDED (Missing: [{missing}])"
                if expected_cls == 1:
                    s2_correct += 1
            else:
                status = "OPEN KRIPKE COUNTERMODEL"
                if expected_cls == 1:
                    s2_correct += 1

        s1_str = f"[{p_f:.2f},{p_u:.2f},{p_t:.2f}] V={float(val_score[0].item()):.2f}"
        target_str = f"Cls {expected_cls}"
        print(f"{name:<34} | {target_str:<7} | {s1_str:<22} | {status}")

    print("=" * 85)
    print(f"System 1 Epistemic Accuracy : {s1_correct}/{len(benchmarks)} ({s1_correct/len(benchmarks)*100:.1f}%)")
    print(f"System 2 Proof & Verification: {s2_correct}/{len(benchmarks)} ({s2_correct/len(benchmarks)*100:.1f}%)")


if __name__ == "__main__":
    main()

