# nanoGentzen-v3: Neuro-Symbolic Typed-Decision Engine (4.93M Params)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-nanoGentzen--v3-yellow)](https://huggingface.co/Sagicc/nanoGentzen-v3)
[![Accuracy](https://img.shields.io/badge/Accuracy-76.75%25%20-success.svg)](#benchmark-results)

**`nanoGentzen-v3`** is a **4.93M parameter (`4,927,812` params) Neuro-Symbolic Formal Decision Engine** that unifies a **Bidirectional RoPE Policy-Value Transformer** with **Gerhard Gentzen (1935) Intuitionistic Sequent Calculus (\(\mathbf{LJ}\))** and **Roy Dyckhoff (1992) Contraction-Free Proof Search (\(\mathbf{G4ip}\))**.

It delivers certified, mathematically explainable decisions across arbitrary JSON states, numeric/regex predicates, natural language policies, and rubric criteria without hallucination:
- **Saul Kripke (1965) 3-Class Epistemic Valuation:** Classifies sequents into `PROVEN_FALSE` (0.0), `INTUITIONISTIC_UNDECIDED` (0.5), and `PROVEN_TRUE` (1.0) in \(< 1.5\text{ ms}\).
- **Intuitionistic Relevance Logic (*IR*) Witness Tracking:** Isolates causal premises actively exercised in derivation from unused distractor facts.
- **Curry-Howard Typed \(\lambda\)-Term Serialization:** Generates verifiable 1-line mathematical certificates (`apply(pair(A, B))`) isomorphic to the verified proof tree DAG.
- **Dual-System Execution:** System 1 fast neural forward pass (~780 decisions/sec, < 1.3 ms latency) + System 2 deductive proof search and post-verification calibration.
- **S1MB Compliant:** Evaluated across all 137 benchmarks in `S1MB-english-v1` (26,269 decisions, 100% valid).
- **Typed Decisions:** Outperforms large language models up to 35B parameters and surpasses the Teacher Consensus Ceiling with **76.75% accuracy** on `LocalLLaMA/typed-decisions` (400 cases, 2,000 decisions).
- **CPU Native:** Runs on standard CPUs (< 150 MB RAM, 0.7 ms – 10.2 ms latency) with zero GPU requirement.

---

## Technical Documentation Suite

For exhaustive theoretical derivations, architectural audits, and the full training methodology, refer to the [`doc/`](doc/README.md) directory:
- [Methodology & Theoretical Foundations](doc/METHODOLOGY.md): Gentzen LJ calculus, Dyckhoff G4ip termination, Saul Kripke relational semantics, Relevance Logic witness tracking, and Curry-Howard isomorphism.
- [Neural Architecture Specifications](doc/ARCHITECTURE.md): Transformer backbone, Span Pointer antecedent attention head (32 slots), Kripke 3-class projection head, and parameter breakdown.
- [Training Workflow & Methodology](doc/TRAINING_WORKFLOW.md): 3-class synthetic corpus curriculum, trajectory-grouped cross-validation, multi-task joint loss supervision, AdamW optimization schedule, and fine-tuning commands.

---

## Benchmark Results

### 1. `LocalLLaMA/typed-decisions` (400 Cases, 2,000 Decisions)

| Model / Architecture | Size | Accuracy ↑ | `noul` Acc ↑ | `choice` Acc ↑ | `score` Acc ↑ | Score Within ±1 ↑ | Score MAE ↓ | KL ↓ | Brier ↓ | ECE ↓ | p50 Latency ↓ | Formal Certificates |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **nanoGentzen-v3** | **4.93M** | **0.7675** | **0.8467** | **0.7533** | **0.7188** | **0.9638** | **0.3143** | **0.2065** | **0.0999** | **0.0728** | **18.27 ms** | **Curry-Howard + IR** |
| *nanoGentzen-Jev-v3 (Pre-Kripke)* | 4.93M | 0.7310 | 0.8283 | 0.6850 | 0.6925 | 0.9725 | 0.2843 | 0.1498 | 0.0809 | 0.0871 | 69.38 ms | Scalar Sigmoid |
| *nanoGentzen-v2 (Grounded)* | 4.93M | 0.7245 | 0.8183 | 0.6767 | 0.6900 | 0.9713 | 0.3191 | 0.1639 | 0.0869 | 0.1268 | 43.65 ms | Binary Value |
| *TypeSafe Jev 1.13.0* | – | 0.7270 | 0.7750 | 0.7200 | 0.6960 | – | – | – | – | 0.1440 | – | Heuristic Rules |
| *Teacher Self-Agreement (Ceiling)* | – | 0.7350 | – | – | – | – | – | – | – | – | – | Soft Consensus |
| *JEV-35B (35B MoE)* | 35B | 0.7000 | – | – | – | – | – | – | – | 0.0620 | > 1,200 ms | Neural LLM |
| *JEV-27B-VL (28B)* | 28B | 0.6880 | – | – | – | – | – | – | – | 0.1130 | > 1,500 ms | Neural LLM |
| *GEV-26B-Decide (26B)* | 26B | 0.5220 | – | – | – | – | – | – | – | 0.0660 | > 900 ms | Neural LLM |
| *JEV-27B (27B)* | 27B | 0.4950 | – | – | – | – | – | – | – | 0.1360 | > 900 ms | Neural LLM |
| *Prior Baseline* | 0M | 0.4700 | – | – | – | – | – | – | – | 0.0880 | < 1 ms | Empirical Prior |
| *THX-01 (0.3B)* | 0.3B | 0.3770 | – | – | – | – | – | – | – | 0.2030 | ~150 ms | Neural LLM |
| *Uniform Random Baseline* | 0M | 0.3080 | – | – | – | – | – | – | – | 0.1690 | < 1 ms | Uniform 1/K |

### 2. S1MB English-v1 Benchmark (137 Benchmarks, 26,269 Decisions)

| Evaluation Scope | Task Avg ↑ | Choice Task ↑ | Noul Task ↑ | Score Task ↑ | Generalization Choice ↑ | Decisions/Sec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **System 2 (Full Neuro-Symbolic Engine)** | **4.42** | **8.63** | **2.12** | **2.50** | **30.06** | ~204 dec/s |
| **System 1 (Pure Neural Epistemic Pass)** | **4.24** | **8.63** | **2.12** | **1.97** | **30.06** | **~778 dec/s** |

- **Score Task (#183 out of 212 on Official Leaderboard):** System 2 achieved **2.50**, outperforming **29 published models**, including `CLM-v0.1-8B` (7.6B params, Score: 1.19), `JevK5-Lite` (437M), `Verdict ModernBERT 151M` (151M), and `LFM2.5-350M-RLCD` (354M).
- **`laya--typed-decisions-score`:** **52.52** (+11.22 points boost from System 2 deductive verification).

---

## Directory Structure

```
.
├── nanogentzen/                    # Core Python package
│   ├── __init__.py                 # Top-level API exports
│   ├── kernel.py                   # Dyckhoff G4ip kernel, verifier & Kripke models
│   ├── search.py                   # System 2 neural-guided proof search
│   ├── modeling_nanogentzen.py     # Transformer + Span Pointer + 3-Class Head
│   ├── configuration_nanogentzen.py# GentzenConfig dataclass & HF config
│   ├── tokenization_nanogentzen.py # HF-compatible LogicTokenizer
│   ├── parser.py                   # Symbolic & natural language rule compiler
│   ├── domain_registry.py          # Enterprise schema compilers
│   ├── dataset.py                  # PyTorch dataset & balanced generator
│   └── vocab.json                  # 95-token logic vocabulary
├── doc/                            # Comprehensive Technical Documentation
│   ├── README.md                   # Documentation index
│   ├── METHODOLOGY.md              # Mathematical & proof-theoretic foundations
│   ├── ARCHITECTURE.md             # Detailed transformer & head specifications
│   └── TRAINING_WORKFLOW.md        # Dataset curriculum, loss formulation & training guide
├── train.py                        # Full training loop (AMP bf16/fp16, AdamW, Cosine LR)
├── dataset.py                      # 3-Class Saul Kripke dataset generator
├── generate_dataset.py             # CLI utility for 3-class dataset synthesis (.pt & .jsonl)
├── jev_server.py                   # FastAPI REST microservice (/v1/decide & /v1/systemone)
├── cli.py                          # Command-line decision runner & REPL shell
├── example_usage.py                # Standalone Python examples
├── test_jev.py                     # API verification suite (6/6 passing)
├── test_pipeline.py                # Integration tests
├── eval_bench.py                   # Hard constructive logic benchmark suite
├── validate_random.py              # 150-sample balanced validation benchmark
├── pt_to_sftnz.py                  # Checkpoint converter (.pt -> model.safetensors)
├── config.json                     # Architecture configuration
├── model.safetensors               # Pretrained FP32/BF16 weights (19.6 MB)
├── pyproject.toml                  # Modern PEP 517/518 build metadata
├── setup.py                        # Standard setuptools configuration
├── requirements.txt                # Package dependencies
└── LICENSE                         # MIT License
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/nanoGentzen/nanoGentzen.git
cd nanoGentzen
pip install -r requirements.txt
pip install -e .
```

Verify the installation:
```bash
python test_jev.py --local
```

### 2. Run Decisions via CLI

```bash
# System 2 Formal Verification
python cli.py --state '{"income": 95000, "credit_score": 740}' --goal approve_loan --rules 'income >= 70000 & credit_score >= 700 => approve_loan'

# Run on CPU with explicit device selection
python cli.py --device cpu --state '{"age": 25}' --goal is_adult --rules 'age >= 18 => is_adult'
```

### 3. Launch REST Microservice

```bash
# Launch server on port 8000 (auto-detects CUDA / CPU)
python jev_server.py --port 8000

# Query with curl:
curl -s -X POST http://127.0.0.1:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nanoGentzen-v3",
    "state": {"is_adult": true, "kyc_passed": true},
    "policy_rules": ["(is_adult & kyc_passed) => approved"],
    "questions": {"approved": {"type": "noul", "target": "approved"}}
  }' | jq .
```

### 4. Training & Fine-Tuning

Generate a balanced 3-class dataset:
```bash
python generate_dataset.py --num-samples 12000 --output data/gentzen_dataset.pt --jsonl data/preview.jsonl
```

Run training with AdamW, mixed precision AMP (BF16), and trajectory-grouped cross-validation:
```bash
python train.py \
    --data-path data/gentzen_dataset.pt \
    --batch-size 128 \
    --epochs 10 \
    --lr 5e-4 \
    --output-checkpoint nanogentzen_checkpoint.pt \
    --output-safetensors model.safetensors
```

Fine-tune from pretrained weights on a domain-specific dataset:
```bash
python train.py \
    --resume-safetensors model.safetensors \
    --data-path domain_dataset.pt \
    --epochs 5 \
    --lr 1e-4 \
    --batch-size 64
```

---

## CPU Execution & Edge Deployment

`nanoGentzen-v3` contains only **4.93M parameters** (~**19.6 MB** weights in `model.safetensors`) and requires **< 150 MB total RAM**:

| Mode | CPU Latency | RAM Usage | GPU Required? |
| :--- | :---: | :---: | :---: |
| **System 1 (Direct Epistemic Forward Pass)** | **0.7 ms – 10.2 ms** | < 150 MB | No |
| **System 2 (Verified Proof Search & Calibration)** | **20 ms – 88 ms** | < 150 MB | No |

---

## Citation

```bibtex
@software{nanogentzen2026,
  author = {nanoGentzen Authors and Contributors},
  title = {nanoGentzen-v3: Neuro-Symbolic Typed-Decision Engine},
  year = {2026},
  url = {https://github.com/nanoGentzen/nanoGentzen}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
