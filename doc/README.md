# nanoGentzen-v3 Documentation Suite

Welcome to the comprehensive technical documentation for **`nanoGentzen-v3`**, a 4.93M parameter Neuro-Symbolic Formal Decision Engine.

This documentation suite details the mathematical foundations, neural architecture, and training workflows powering the engine.

---

## Documentation Index

### 1. [Methodology & Theoretical Foundations](METHODOLOGY.md)
Detailed logical and mathematical foundations:
- **Gerhard Gentzen (1935) Intuitionistic Sequent Calculus (\(\mathbf{LJ}\)):** Constructive derivation rules and sequent structure.
- **Roy Dyckhoff (1992) Contraction-Free Sequent Calculus (\(\mathbf{G4ip}\)):** Deterministic termination without structural contraction cycles.
- **Saul Kripke (1965) 3-Class Epistemic Semantics:** Mathematical frame \(\mathcal{M} = \langle W, \le, \Vdash \rangle\), epistemic states (`PROVEN_TRUE`, `INTUITIONISTIC_UNDECIDED`, `PROVEN_FALSE`).
- **Intuitionistic Relevance Logic (*IR*) Witness Tracking:** Causal leaves vs distractor premises, relevance ratio calculation.
- **Curry-Howard Typed \(\lambda\)-Calculus Isomorphism:** Proof-tree isomorphism to compact, externally verifiable \(\lambda\)-terms.

### 2. [Neural Architecture Specifications](ARCHITECTURE.md)
Model architecture, layer dimensions, and specialized prediction heads:
- **Transformer Backbone:** 6 bidirectional transformer blocks, RoPE rotary embeddings, 256 hidden dimension, 8 attention heads.
- **Boundary-Aware Span Pointer Head:** Scaled dot-product antecedent pooling across 32 slots for left-rule pivot selection.
- **Kripke 3-Class Epistemic Valuation Head:** Softmax distribution over `[False, Undecided, True]` and epistemic sharpness metric.
- **Dual-System Execution:** System 1 fast neural forward pass (< 1.5 ms) vs System 2 verified constructive proof search (10–25 ms).
- **Parameter Breakdown:** Exact parameter allocation (4,927,812 parameters total, 19.6 MB safetensors).

### 3. [Training Workflow & Methodology](TRAINING_WORKFLOW.md)
End-to-end data generation, optimization, and training guide:
- **Synthetic Corpus Curriculum:** 1:1:1 balanced generation across constructive proofs, missing premise fallacies, and refutations.
- **Trajectory-Grouped Validation:** Zero proof-tree data leakage between training and validation splits.
- **Multi-Task Joint Loss:** Unified objective combining Rule CE, Span Pointer CE, Kripke Epistemic CE, and Continuous Value regression.
- **Optimization Strategy:** AdamW decoupled weight decay, warmup + cosine annealing schedule, mixed precision (AMP BF16/FP16) with FP32 master weights.
- **CLI Commands:** Complete instructions for dataset synthesis, scratch training, fine-tuning, and evaluation.

---

## Quick Reference: Core Metrics & Capabilities

| Capability | System 1 (Epistemic Pass) | System 2 (Verified Proof Search) |
| :--- | :---: | :---: |
| **Primary Method** | `model.system_one(req)` | `model.decide(req)` |
| **Typical Latency (GPU)** | `0.9 ms – 1.4 ms` | `10 ms – 25 ms` |
| **Typical Latency (CPU)** | `0.7 ms – 10.2 ms` | `20 ms – 88 ms` |
| **Throughput** | ~`780 decisions/sec` | ~`200 decisions/sec` |
| **Soundness** | Statistical / Epistemic Soft Prior | **100% Mathematically Certified** |
| **Audit Artifacts** | Attention Weights & Probabilities | Proof Tree DAG, Curry-Howard Term, *IR* Witnesses |
| **Failure Diagnosis** | Low Epistemic Sharpness | Concrete 2-World Kripke Countermodel |

