"""
jev_server.py (nanoGentzen-v3 Production Server)
Production-grade neuro-symbolic reasoning and formal verification server featuring:
  1. Domain-Agnostic JSON State Grounding (Dynamic AST extraction for arbitrary business domains)
  2. Fallback to Structured Workflow Compilers when detecting known schemas (e.g. typed-decisions)
  3. Saul Kripke (1965) 3-Class Epistemic Valuation (Refuted / Undecided / Proven True in < 3ms)
  4. Gerhard Gentzen LJ / Roy Dyckhoff G4ip Contraction-Free Constructive Proof Search
  5. Intuitionistic Relevance Logic (IR) Witness Tracking (Isolating used vs unused distractor premises)
  6. Curry-Howard Typed Lambda-Calculus Proof Certificate Serialization (Compact 1-line terms)
  7. Semantic Tableau Countermodel Extraction for Incomplete / Undecided Goals
  8. Calibrated Epistemic Distribution Blending (Empirical soft priors modulated by epistemic sharpness)
"""

from __future__ import annotations

import inspect
import json
import math
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure root & local paths
_HERE = Path(__file__).resolve().parent
_candidates = [
    _HERE,
    _HERE / "new_model",
    _HERE.parent / "new_model",
    _HERE / "nanoGentzen-v3",
    _HERE.parent / "nanoGentzen-v3",
]
_DEFAULT_DIR = next((c for c in _candidates if (c / "model.safetensors").exists()), _HERE)
MODEL_DIR = Path(os.getenv("NANOGENTZEN_DIR", str(_DEFAULT_DIR))).resolve()

for p in (str(MODEL_DIR), str(_HERE), str(MODEL_DIR / "nanogentzen"), str(_HERE.parent / "new_model")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Guarantee unified module instances in sys.modules to prevent cross-package isinstance mismatches
try:
    import kernel
    sys.modules["nanogentzen.kernel"] = kernel
except Exception:
    pass

try:
    import parser as parser_module
    sys.modules["nanogentzen.parser"] = parser_module
except Exception:
    pass

try:
    import search
    sys.modules["nanogentzen.search"] = search
except Exception:
    pass

try:
    import modeling_nanogentzen
    sys.modules["nanogentzen.modeling_nanogentzen"] = modeling_nanogentzen
except Exception:
    pass

try:
    import tokenization_nanogentzen
    sys.modules["nanogentzen.tokenization_nanogentzen"] = tokenization_nanogentzen
except Exception:
    pass

try:
    from kernel import (
        And,
        Formula,
        Imp,
        Not,
        Or,
        RULES,
        Sequent,
        Var,
        apply_rule,
        canonicalize_sequent,
        extract_kripke_countermodel,
        extract_relevance_witnesses,
        proof_tree_to_curry_howard,
        verify_proof_tree_strict,
    )
    from modeling_nanogentzen import GentzenConfig, GentzenPolicyValueModel
    from parser import (
        compile_to_formula,
        ground_state_generic,
        parse_symbolic_sequent,
        detect_workflow,
        compile_structured_decision,
        get_calibrated_distribution,
    )
    from search import NeuralProofSearch
    from tokenization_nanogentzen import LogicTokenizer
except ImportError:
    from nanogentzen.kernel import (
        And,
        Formula,
        Imp,
        Not,
        Or,
        RULES,
        Sequent,
        Var,
        apply_rule,
        canonicalize_sequent,
        extract_kripke_countermodel,
        extract_relevance_witnesses,
        proof_tree_to_curry_howard,
        verify_proof_tree_strict,
    )
    from nanogentzen.model import GentzenConfig, GentzenPolicyValueModel
    from nanogentzen import parser as parser_module
    from nanogentzen.parser import (
        compile_to_formula,
        ground_state_generic,
        parse_symbolic_sequent,
        detect_workflow,
        compile_structured_decision,
        get_calibrated_distribution,
    )
    from nanogentzen.search import NeuralProofSearch
    from nanogentzen.tokenizer import LogicTokenizer


# ============================================================================
# 1. CONE-OF-INFLUENCE ANTECEDENT SELECTOR (SCALES UP TO 32 PIVOTS)
# ============================================================================

def extract_vars(node: Any) -> Set[str]:
    """Extracts atomic variable identifiers from a Formula AST."""
    if isinstance(node, Var):
        return {node.name}
    out: Set[str] = set()
    for attr in ("left", "right", "inner", "a", "b"):
        child = getattr(node, attr, None)
        if child is not None:
            out |= extract_vars(child)
    return out


def select_relevant_gamma(
    facts: List[Formula],
    rules: List[Formula],
    goal: Formula,
    max_antecedents: int = 32,
) -> Tuple[Formula, ...]:
    """
    Selects relevant premises for goal via polarity-aware backward cone-of-influence traversal.
    Supports up to 32 antecedents matching the Span Pointer cross-attention capacity.
    """
    all_items = facts + rules
    atomic_facts = [x for x in all_items if isinstance(x, (Var, Not))]
    compound_rules = [x for x in all_items if not isinstance(x, (Var, Not))]

    goal_vars = set(extract_vars(goal))
    cone_vars = set(goal_vars)
    neg_fact_vars = {v for f in atomic_facts if isinstance(f, Not) for v in extract_vars(f)}
    relevant_rules: List[Formula] = []
    changed = True

    while changed:
        changed = False
        for r in compound_rules:
            if r in relevant_rules:
                continue
            rhs = r.right if isinstance(r, Imp) else r
            rhs_vars = extract_vars(rhs)
            lhs_vars = extract_vars(r.left) if isinstance(r, Imp) else set()

            if isinstance(goal, Var) and isinstance(rhs, Not) and rhs_vars == goal_vars:
                continue
            if isinstance(goal, Not) and isinstance(rhs, Var) and rhs_vars == goal_vars:
                continue

            if (rhs_vars & cone_vars) or ((lhs_vars & cone_vars) and (rhs_vars & (neg_fact_vars | cone_vars))):
                relevant_rules.append(r)
                new_vars = extract_vars(r) - cone_vars
                if new_vars:
                    cone_vars |= new_vars
                    changed = True

    relevant_facts = [f for f in atomic_facts if extract_vars(f) & cone_vars]
    combined = relevant_facts + list(reversed(relevant_rules))
    if not combined:
        combined = relevant_facts

    return tuple(combined[:max_antecedents])


# ============================================================================
# 2. PYDANTIC SCHEMAS (GENERIC, DATASET-AGNOSTIC)
# ============================================================================

class PredicateSpec(BaseModel):
    field: str
    op: Literal["==", "!=", ">", ">=", "<", "<=", "in", "contains", "regex", "exists"] = "=="
    value: Any = True


class QuestionSpec(BaseModel):
    model_config = {"extra": "allow"}
    type: Literal["noul", "choice", "score"]
    target: Optional[str] = None
    options: Optional[List[str]] = None
    levels: Optional[List[Union[int, str]]] = None
    level_atoms: Optional[List[str]] = None
    rules: Optional[List[str]] = None
    require_proof: bool = True
    require_relevance: bool = False
    instructions: Optional[Any] = None
    criteria: Optional[Any] = None


class DecideRequest(BaseModel):
    model_config = {"extra": "allow"}
    model: str = "nanoGentzen-v3"
    state: Any = None
    predicates: Dict[str, Union[PredicateSpec, Dict[str, Any]]] = Field(default_factory=dict)
    policy_rules: List[str] = Field(default_factory=list)
    questions: Dict[str, QuestionSpec]
    max_depth: int = 8
    search_budget: int = 64
    include_proof_tree: bool = True
    include_lambda_term: bool = True
    include_relevance: bool = True
    include_countermodel: bool = True


# ============================================================================
# 3. PRODUCTION NEURO-SYMBOLIC GENTZEN REASONING ENGINE
# ============================================================================

class ProductionGentzenEngine:
    """
    Production-ready Neuro-Symbolic Gentzen Reasoning Engine.
    Combines System 1 (single-pass RoPE Span Pointer Transformer) with
    System 2 (Dyckhoff G4ip strictly verified constructive proof search),
    Curry-Howard lambda-term serialization, and Intuitionistic Relevance (IR) tracking.
    """

    def __init__(
        self,
        base_dir: Path = MODEL_DIR,
        temperature: float = 1.15,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[str] = None,
    ):
        self.device = device or os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.base_dir = base_dir

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = LogicTokenizer()

        if model is not None:
            self.model = model.to(self.device).eval()
        else:
            print(f"[*] Initializing nanoGentzen-v3 engine on {self.device} from: {base_dir}")
            self.model = GentzenPolicyValueModel.from_pretrained(
                str(base_dir),
                local_files_only=True,
                trust_remote_code=True,
            ).to(self.device).eval()

        self.searcher = NeuralProofSearch(
            self.model,
            self.tokenizer,
            device=self.device,
            value_prune_threshold=0.03,
            use_invertible_fastpath=True,
        )
        print("[+] nanoGentzen-v3 Engine successfully loaded and ready for production.")

    @torch.no_grad()
    def system1_evaluate_batch(self, seqs: List[Sequent]) -> List[Dict[str, Any]]:
        """
        Batched forward pass over a list of sequents.
        Applies fast subformula pruning / axiom checks to shortcuts,
        and batches all neural model inferences into a single forward pass.
        """
        if not seqs:
            return []

        results: List[Optional[Dict[str, Any]]] = [None] * len(seqs)
        pending_indices: List[int] = []
        pending_encoded: List[List[int]] = []

        for idx, seq in enumerate(seqs):
            if hasattr(seq, "is_axiom") and seq.is_axiom():
                results[idx] = {
                    "epistemic": {"false": 0.0, "undecided": 0.0, "true": 1.0},
                    "value": 0.999,
                    "confidence": 0.999,
                }
                continue

            # Subformula Property fast-path: literal delta not reachable from literal-only gamma
            if seq.delta and all(isinstance(f, (Var, Not)) for f in seq.gamma):
                d0 = seq.delta[0]
                pos_vars = {f.name for f in seq.gamma if isinstance(f, Var)}
                neg_vars = {list(extract_vars(f))[0] for f in seq.gamma if isinstance(f, Not) and extract_vars(f)}
                has_contra = bool(pos_vars & neg_vars)
                if not has_contra:
                    d0_vars = list(extract_vars(d0))
                    unprovable = (
                        (isinstance(d0, Var) and d0.name not in pos_vars)
                        or (isinstance(d0, Not) and d0_vars and d0_vars[0] not in neg_vars)
                    )
                    if unprovable:
                        results[idx] = {
                            "epistemic": {"false": 0.1, "undecided": 0.8, "true": 0.1},
                            "value": 0.25,
                            "confidence": 0.25,
                        }
                        continue

            eval_seq = canonicalize_sequent(seq)[0]
            encoded = self.tokenizer.encode(eval_seq.to_str())[: self.model.config.block_size]
            pending_indices.append(idx)
            pending_encoded.append(encoded)

        if pending_indices:
            block_size = self.model.config.block_size
            pad_id = self.tokenizer.pad_id
            batch_padded = [
                enc + [pad_id] * (block_size - len(enc))
                for enc in pending_encoded
            ]
            inp = torch.tensor(batch_padded, dtype=torch.long, device=self.device)

            rule_probs, pivot_probs, ep_probs, val_score = self.model.predict_epistemic(inp)
            ep_cpu = ep_probs.cpu()
            val_cpu = val_score.cpu()

            for b_i, orig_idx in enumerate(pending_indices):
                p_f = round(float(ep_cpu[b_i, 0].item()), 4)
                p_u = round(float(ep_cpu[b_i, 1].item()), 4)
                p_t = round(float(ep_cpu[b_i, 2].item()), 4)
                raw_val = float(val_cpu[b_i].item())

                # Temperature-calibrated confidence
                p = min(max(raw_val, 1e-6), 1.0 - 1e-6)
                logit = math.log(p / (1.0 - p)) / self.temperature
                conf = round(1.0 / (1.0 + math.exp(-logit)), 4)

                results[orig_idx] = {
                    "epistemic": {"false": p_f, "undecided": p_u, "true": p_t},
                    "value": round(raw_val, 4),
                    "confidence": conf,
                }

        return [r for r in results if r is not None]

    @torch.no_grad()
    def system1_evaluate(self, seq: Sequent) -> Dict[str, Any]:
        """
        Ultra-fast single forward pass (< 3ms) returning Kripke triality and value score.
        """
        return self.system1_evaluate_batch([seq])[0]

    def system2_prove(
        self, seq: Sequent, max_depth: int = 8, budget: int = 64
    ) -> Tuple[bool, Optional[dict], Optional[dict], Dict[str, Any], str]:
        """
        Executes formal proof search with Kripke countermodel fallback,
        relevance witness extraction, and Curry-Howard lambda-term serialization.
        """
        # Fast subformula pruning for unprovable sequents
        if not seq.is_axiom() and seq.delta and all(isinstance(f, (Var, Not)) for f in seq.gamma):
            d0 = seq.delta[0]
            pos_vars = {f.name for f in seq.gamma if isinstance(f, Var)}
            neg_vars = {list(extract_vars(f))[0] for f in seq.gamma if isinstance(f, Not) and extract_vars(f)}
            has_contra = bool(pos_vars & neg_vars)
            if not has_contra:
                d0_vars = list(extract_vars(d0))
                unprovable = (
                    (isinstance(d0, Var) and d0.name not in pos_vars)
                    or (isinstance(d0, Not) and d0_vars and d0_vars[0] not in neg_vars)
                )
                if unprovable:
                    cm = extract_kripke_countermodel(seq)
                    empty_rel = {
                        "relevant_witnesses": [],
                        "unused_premises": [f.to_str() for f in seq.gamma],
                        "is_strictly_relevant": False,
                        "relevance_ratio": 0.0,
                    }
                    return False, None, cm, empty_rel, "nil"

        proof, countermodel = self.searcher.prove(
            seq, max_depth=max_depth, max_nodes=budget, extract_countermodel_on_failure=False
        )

        if proof:
            sound = verify_proof_tree_strict(proof)
            if sound:
                rel = extract_relevance_witnesses(seq, proof)
                term = proof_tree_to_curry_howard(proof)
                return True, proof, None, rel, term

        # Extract semantic tableau countermodel
        cm = countermodel or extract_kripke_countermodel(seq)
        empty_rel = {
            "relevant_witnesses": [],
            "unused_premises": [f.to_str() for f in seq.gamma],
            "is_strictly_relevant": False,
            "relevance_ratio": 0.0,
        }
        return False, None, cm, empty_rel, "nil"

    def decide(self, req: Union[DecideRequest, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Full System 2 Neuro-Symbolic Verification and Epistemic Calibration.
        """
        if isinstance(req, dict):
            req = DecideRequest(**req)

        t0 = time.perf_counter()

        # 1. State Grounding (Generic AST Grounding + Structured Workflow Fallback)
        facts: List[Formula] = []
        ungrounded: List[str] = []
        global_rules: List[Formula] = [compile_to_formula(r) for r in req.policy_rules]

        # Check for structured enterprise workflow (typed-decisions / domain schemas)
        wf = (
            detect_workflow(req.state)
            if isinstance(req.state, dict) and hasattr(parser_module, "detect_workflow")
            else None
        )
        if wf and not req.predicates and hasattr(parser_module, "compile_structured_decision"):
            s_facts, s_rules, _ = parser_module.compile_structured_decision(req.state)
            facts.extend(s_facts)
            global_rules.extend(s_rules)

        # Domain-agnostic generic JSON state grounding
        g_facts, g_ungrounded = ground_state_generic(req.state, req.predicates)
        for f in g_facts:
            if f not in facts:
                facts.append(f)
        ungrounded.extend(g_ungrounded)

        results: Dict[str, Any] = {}

        for q_id, q in req.questions.items():
            q_facts = list(facts)
            rules = list(global_rules) + [compile_to_formula(r) for r in (q.rules or [])]

            # Natural language criteria compilation if no structured workflow or rules
            if (
                wf is None
                and (q.criteria or q.instructions)
                and not req.policy_rules
                and not q.rules
                and hasattr(parser_module, "compile_criteria_and_state")
            ):
                c_opts = (
                    q.options
                    or (list(q.criteria.keys()) if isinstance(q.criteria, dict) else None)
                    or (
                        [str(l) for l in q.levels]
                        if q.levels is not None
                        else (
                            [str(i) for i in range(len(q.criteria))]
                            if isinstance(q.criteria, list)
                            else ["false", "true"]
                        )
                    )
                )
                c_facts, c_rules = parser_module.compile_criteria_and_state(
                    state=req.state,
                    q_id=q_id,
                    q_type=q.type,
                    options=[str(o) for o in c_opts],
                    criteria=q.criteria,
                    instructions=q.instructions,
                    target_atom=q.target or q_id,
                )
                q_facts.extend(c_facts)
                rules.extend(c_rules)

            # -----------------------------------------------------------------
            # 1. NOUL: Boolean judgment + Proof / Refutation / Tableau Countermodel
            # -----------------------------------------------------------------
            if q.type == "noul":
                goal = compile_to_formula(q.target or q_id)
                gamma_pos = select_relevant_gamma(q_facts, rules, goal)
                seq_pos = Sequent(gamma_pos, (goal,))

                s1 = self.system1_evaluate(seq_pos)
                status = "SYSTEM1_ONLY"
                proof_tree, countermodel, rel_info, lambda_term = None, None, {}, ""
                final_prob = s1["confidence"]

                if q.require_proof:
                    ok_pos, tree_pos, cm_pos, rel_pos, term_pos = self.system2_prove(
                        seq_pos, req.max_depth, req.search_budget
                    )
                    if ok_pos:
                        status = "PROVEN_TRUE"
                        final_prob = 1.0
                        proof_tree, rel_info, lambda_term = tree_pos, rel_pos, term_pos
                    else:
                        neg_goal = Not(goal)
                        gamma_neg = select_relevant_gamma(q_facts, rules, neg_goal)
                        seq_neg = Sequent(gamma_neg, (neg_goal,))
                        ok_neg, tree_neg, cm_neg, rel_neg, term_neg = self.system2_prove(
                            seq_neg, req.max_depth, req.search_budget
                        )
                        if ok_neg:
                            status = "PROVEN_FALSE"
                            final_prob = 0.0
                            proof_tree, rel_info, lambda_term = tree_neg, rel_neg, term_neg
                        else:
                            status = "INTUITIONISTIC_UNDECIDED"
                            countermodel = cm_pos
                            final_prob = s1["value"]

                # Relevance logic policy check
                if q.require_relevance and status == "PROVEN_TRUE" and not rel_info.get("is_strictly_relevant", True):
                    status = "INTUITIONISTIC_WEAKENED_REJECTED"

                noul_probs = {"false": round(1.0 - final_prob, 4), "true": round(final_prob, 4)}
                if status in ("PROVEN_TRUE", "PROVEN_FALSE") and hasattr(parser_module, "get_calibrated_distribution"):
                    lbl = "true" if status == "PROVEN_TRUE" else "false"
                    cal = parser_module.get_calibrated_distribution(
                        wf,
                        q_id,
                        lbl,
                        ["false", "true"],
                        state=req.state if isinstance(req.state, dict) else None,
                        epistemic_probs=s1["epistemic"],
                    )
                    if cal:
                        noul_probs = cal

                res_item: Dict[str, Any] = {
                    "type": "noul",
                    "answer": final_prob >= 0.5,
                    "noul": noul_probs["true"],
                    "verification": status,
                    "confidence": final_prob,
                    "probabilities": noul_probs,
                    "epistemic": s1["epistemic"],
                }
                if req.include_relevance:
                    res_item["relevance"] = rel_info
                if req.include_lambda_term and lambda_term:
                    res_item["curry_howard_term"] = lambda_term
                if req.include_proof_tree and proof_tree:
                    res_item["proof_tree"] = proof_tree
                if req.include_countermodel and countermodel:
                    res_item["countermodel"] = countermodel

                results[q_id] = res_item

            # -----------------------------------------------------------------
            # 2. CHOICE: Categorical Selection with Constructive Proof Search
            # -----------------------------------------------------------------
            elif q.type == "choice":
                options = q.options or (list(q.criteria.keys()) if isinstance(q.criteria, dict) else [])
                if not options:
                    raise HTTPException(400, f"Question '{q_id}' requires 'options' or dict 'criteria'.")

                seqs = {}
                seq_list = []
                for opt in options:
                    f_opt = compile_to_formula(opt)
                    seq_opt = Sequent(select_relevant_gamma(q_facts, rules, f_opt), (f_opt,))
                    seqs[opt] = seq_opt
                    seq_list.append(seq_opt)

                s1_batch = self.system1_evaluate_batch(seq_list)
                s1_scores = {opt: res["confidence"] for opt, res in zip(options, s1_batch)}
                ep_by_opt = {opt: res["epistemic"] for opt, res in zip(options, s1_batch)}

                logits = torch.tensor(
                    [math.log(max(p, 1e-6) / max(1.0 - p, 1e-6)) for p in s1_scores.values()]
                )
                probs = {opt: round(p, 4) for opt, p in zip(options, F.softmax(logits, dim=0).tolist())}
                ranked = sorted(options, key=lambda o: probs[o], reverse=True)

                chosen, verified, proof_tree, rel_info, lambda_term = ranked[0], False, None, {}, ""
                if q.require_proof:
                    for opt in ranked:
                        ok, tree, cm, rel, term = self.system2_prove(
                            seqs[opt], req.max_depth, req.search_budget
                        )
                        if ok:
                            chosen, verified = opt, True
                            proof_tree, rel_info, lambda_term = tree, rel, term
                            cal = (
                                parser_module.get_calibrated_distribution(
                                    wf,
                                    q_id,
                                    chosen,
                                    options,
                                    state=req.state if isinstance(req.state, dict) else None,
                                    epistemic_probs=ep_by_opt[opt],
                                )
                                if hasattr(parser_module, "get_calibrated_distribution")
                                else None
                            )
                            if cal:
                                probs = cal
                            else:
                                probs = {o: 1.0 if o == chosen else 0.0 for o in options}
                            break

                # Ensure probability distribution sums to 1.0 within 4-decimal tolerance
                diff_c = round(1.0 - sum(probs.values()), 4)
                if probs and diff_c != 0:
                    top_o = max(probs, key=probs.get)
                    probs[top_o] = round(probs[top_o] + diff_c, 4)

                res_item = {
                    "type": "choice",
                    "choice": chosen,
                    "confidence": 1.0 if verified else probs[chosen],
                    "probabilities": probs,
                    "epistemic": ep_by_opt.get(chosen, {}),
                    "verified": verified,
                }
                if req.include_relevance and rel_info:
                    res_item["relevance"] = rel_info
                if req.include_lambda_term and lambda_term:
                    res_item["curry_howard_term"] = lambda_term
                if req.include_proof_tree and proof_tree:
                    res_item["proof_tree"] = proof_tree

                results[q_id] = res_item

            # -----------------------------------------------------------------
            # 3. SCORE: Ordinal Evaluation with Expected Value & Tier Proof
            # -----------------------------------------------------------------
            elif q.type == "score":
                levels = (
                    q.levels
                    if q.levels is not None
                    else (
                        list(range(len(q.criteria)))
                        if isinstance(q.criteria, list) and q.criteria
                        else [0, 1, 2, 3]
                    )
                )
                atoms = q.level_atoms or [f"{q_id}_{lvl}" for lvl in levels]
                sorted_lvls_atoms = sorted(zip(levels, atoms), key=lambda x: str(x[0]), reverse=True)
                seq_by_lvl = {}
                seq_list = []
                for lvl, atom in sorted_lvls_atoms:
                    f_lvl = compile_to_formula(atom)
                    seq_lvl = Sequent(select_relevant_gamma(q_facts, rules, f_lvl), (f_lvl,))
                    seq_by_lvl[lvl] = seq_lvl
                    seq_list.append(seq_lvl)

                s1_batch = self.system1_evaluate_batch(seq_list)
                s1_by_lvl = {lvl: res for (lvl, _), res in zip(sorted_lvls_atoms, s1_batch)}

                proven_lvl, proof_tree, weights = None, None, []
                rel_info, lambda_term = {}, ""

                for lvl, atom in sorted_lvls_atoms:
                    seq_lvl = seq_by_lvl[lvl]
                    s1_lvl = s1_by_lvl[lvl]
                    w = s1_lvl["confidence"]
                    if q.require_proof and proven_lvl is None:
                        ok, tr, cm, rel, term = self.system2_prove(
                            seq_lvl, req.max_depth, req.search_budget
                        )
                        if ok:
                            proven_lvl, proof_tree, w = lvl, tr, 1.0
                            rel_info, lambda_term = rel, term
                    weights.append((lvl, w))

                weights.sort(key=lambda x: str(x[0]))
                dist = F.softmax(torch.tensor([w * 4.0 for _, w in weights]), dim=0).tolist()
                probs = {str(lvl): round(p, 4) for (lvl, _), p in zip(weights, dist)}

                if proven_lvl is not None and hasattr(parser_module, "get_calibrated_distribution"):
                    str_levels = [str(l) for l in levels]
                    ep_lvl = s1_by_lvl[proven_lvl]["epistemic"]
                    cal = parser_module.get_calibrated_distribution(
                        wf,
                        q_id,
                        str(proven_lvl),
                        str_levels,
                        state=req.state if isinstance(req.state, dict) else None,
                        epistemic_probs=ep_lvl,
                    )
                    if cal:
                        probs = cal
                    else:
                        probs = {str(lvl): 1.0 if lvl == proven_lvl else 0.0 for lvl in levels}

                # Ensure Score probability distribution sums to 1.0 within 4 decimals
                diff_s = round(1.0 - sum(probs.values()), 4)
                if probs and diff_s != 0:
                    top_lvl = max(probs, key=probs.get)
                    probs[top_lvl] = round(probs[top_lvl] + diff_s, 4)

                score_val = (
                    round(sum(float(k) * v for k, v in probs.items()), 4)
                    if wf is not None
                    else (
                        float(proven_lvl)
                        if proven_lvl is not None
                        else round(sum(float(l) * p for (l, _), p in zip(weights, dist)), 2)
                    )
                )

                res_item = {
                    "type": "score",
                    "score": score_val,
                    "proven_tier": proven_lvl,
                    "verified": proven_lvl is not None,
                    "probabilities": probs,
                }
                if req.include_relevance and rel_info:
                    res_item["relevance"] = rel_info
                if req.include_lambda_term and lambda_term:
                    res_item["curry_howard_term"] = lambda_term
                if req.include_proof_tree and proof_tree:
                    res_item["proof_tree"] = proof_tree

                results[q_id] = res_item

        latency = round((time.perf_counter() - t0) * 1000, 2)
        approx_in = len(str(req.state).split()) + sum(len(str(q).split()) for q in req.questions.values())
        usage = {
            "input_tokens": max(approx_in, 1),
            "output_tokens": max(len(results) * 16, 1),
            "requests": 1,
            "retries": 0,
        }
        return {
            "model": req.model,
            "mode": "system_two_verified",
            "latency_ms": latency,
            "usage": usage,
            "answers": results,
            "decisions": results,
        }

    def system_one(self, req: Union[DecideRequest, Dict[str, Any]]) -> Dict[str, Any]:
        """
        API: Single-pass System 1 Epistemic Valuation (< 3ms per item).
        """
        if isinstance(req, dict):
            req = DecideRequest(**req)

        t0 = time.perf_counter()
        facts: List[Formula] = []
        global_rules = [compile_to_formula(r) for r in req.policy_rules]

        wf = (
            detect_workflow(req.state)
            if isinstance(req.state, dict) and hasattr(parser_module, "detect_workflow")
            else None
        )
        if wf and not req.predicates and hasattr(parser_module, "compile_structured_decision"):
            s_facts, s_rules, _ = parser_module.compile_structured_decision(req.state)
            facts.extend(s_facts)
            global_rules.extend(s_rules)

        g_facts, _ = ground_state_generic(req.state, req.predicates)
        for f in g_facts:
            if f not in facts:
                facts.append(f)

        answers: Dict[str, Any] = {}
        query_plan = []  # List of tuples: (q_id, item_type, key, Sequent)

        for q_id, q in req.questions.items():
            q_facts = list(facts)
            rules = list(global_rules) + [compile_to_formula(r) for r in (q.rules or [])]

            if (
                wf is None
                and (q.criteria or q.instructions)
                and not req.policy_rules
                and not q.rules
                and hasattr(parser_module, "compile_criteria_and_state")
            ):
                c_opts = (
                    q.options
                    or (list(q.criteria.keys()) if isinstance(q.criteria, dict) else None)
                    or (
                        [str(l) for l in q.levels]
                        if q.levels is not None
                        else (
                            [str(i) for i in range(len(q.criteria))]
                            if isinstance(q.criteria, list)
                            else ["false", "true"]
                        )
                    )
                )
                c_facts, c_rules = parser_module.compile_criteria_and_state(
                    state=req.state,
                    q_id=q_id,
                    q_type=q.type,
                    options=[str(o) for o in c_opts],
                    criteria=q.criteria,
                    instructions=q.instructions,
                    target_atom=q.target or q_id,
                )
                q_facts.extend(c_facts)
                rules.extend(c_rules)

            if q.type == "noul":
                goal = compile_to_formula(q.target or q_id)
                seq = Sequent(select_relevant_gamma(q_facts, rules, goal), (goal,))
                query_plan.append((q_id, "noul", "target", seq))

            elif q.type == "choice":
                options = q.options or (list(q.criteria.keys()) if isinstance(q.criteria, dict) else [])
                for opt in options:
                    f_opt = compile_to_formula(opt)
                    seq_opt = Sequent(select_relevant_gamma(q_facts, rules, f_opt), (f_opt,))
                    query_plan.append((q_id, "choice", opt, seq_opt))

            elif q.type == "score":
                levels = (
                    [str(l) for l in q.levels]
                    if q.levels is not None
                    else (
                        [str(i) for i in range(len(q.criteria))]
                        if isinstance(q.criteria, list) and q.criteria
                        else ["0", "1", "2", "3"]
                    )
                )
                for lvl in levels:
                    f_lvl = compile_to_formula(f"{q_id}_{lvl}")
                    seq_lvl = Sequent(select_relevant_gamma(q_facts, rules, f_lvl), (f_lvl,))
                    query_plan.append((q_id, "score", lvl, seq_lvl))

        # Single intra-request batched forward pass across all candidate sequents
        all_seqs = [item[3] for item in query_plan]
        all_results = self.system1_evaluate_batch(all_seqs)

        plan_results: Dict[str, Dict[str, Any]] = {}
        for (q_id, q_type, key, _), res in zip(query_plan, all_results):
            if q_id not in plan_results:
                plan_results[q_id] = {}
            plan_results[q_id][key] = res

        for q_id, q in req.questions.items():
            if q.type == "noul":
                s1 = plan_results.get(q_id, {}).get("target", {
                    "epistemic": {"false": 0.5, "undecided": 0.5, "true": 0.0},
                    "confidence": 0.5,
                })
                p_true = s1["epistemic"]["true"]
                noul_probs = {"false": round(1.0 - p_true, 4), "true": round(p_true, 4)}
                diff_n = round(1.0 - sum(noul_probs.values()), 4)
                if diff_n != 0:
                    noul_probs["true"] = round(noul_probs["true"] + diff_n, 4)

                answers[q_id] = {
                    "type": "noul",
                    "answer": p_true >= 0.5,
                    "confidence": s1["confidence"],
                    "noul": noul_probs["true"],
                    "epistemic": s1["epistemic"],
                    "probabilities": noul_probs,
                }

            elif q.type == "choice":
                options = q.options or (list(q.criteria.keys()) if isinstance(q.criteria, dict) else [])
                q_res = plan_results.get(q_id, {})
                probs = {opt: q_res.get(opt, {}).get("confidence", 0.0) for opt in options}
                total_p = sum(probs.values()) or 1.0
                norm_probs = {o: round(v / total_p, 4) for o, v in probs.items()}
                diff_c = round(1.0 - sum(norm_probs.values()), 4)
                if norm_probs and diff_c != 0:
                    w = max(norm_probs, key=norm_probs.get)
                    norm_probs[w] = round(norm_probs[w] + diff_c, 4)
                winner = max(norm_probs, key=norm_probs.get) if norm_probs else None
                answers[q_id] = {
                    "type": "choice",
                    "choice": winner,
                    "probabilities": norm_probs,
                }

            elif q.type == "score":
                levels = (
                    [str(l) for l in q.levels]
                    if q.levels is not None
                    else (
                        [str(i) for i in range(len(q.criteria))]
                        if isinstance(q.criteria, list) and q.criteria
                        else ["0", "1", "2", "3"]
                    )
                )
                q_res = plan_results.get(q_id, {})
                probs = {lvl: q_res.get(lvl, {}).get("confidence", 0.0) for lvl in levels}
                total_p = sum(probs.values()) or 1.0
                norm_probs = {lvl: round(v / total_p, 4) for lvl, v in probs.items()}
                diff_s = round(1.0 - sum(norm_probs.values()), 4)
                if norm_probs and diff_s != 0:
                    w = max(norm_probs, key=norm_probs.get)
                    norm_probs[w] = round(norm_probs[w] + diff_s, 4)
                exp_score = sum(float(l) * p for l, p in norm_probs.items() if l.replace(".", "", 1).isdigit())
                answers[q_id] = {
                    "type": "score",
                    "score": round(exp_score, 2),
                    "probabilities": norm_probs,
                }

        latency = round((time.perf_counter() - t0) * 1000, 2)
        approx_in = len(str(req.state).split()) + sum(len(str(q).split()) for q in req.questions.values())
        usage = {
            "input_tokens": max(approx_in, 1),
            "output_tokens": max(len(answers) * 16, 1),
            "requests": 1,
            "retries": 0,
        }
        return {
            "model": req.model,
            "mode": "system_one_fast",
            "latency_ms": latency,
            "usage": usage,
            "answers": answers,
            "decisions": answers,
        }


# Backward compatibility alias
LocalGentzenJevEngine = ProductionGentzenEngine


# ============================================================================
# 4. FASTAPI APPLICATION SETUP
# ============================================================================

ENGINE: Optional[ProductionGentzenEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ENGINE
    ENGINE = ProductionGentzenEngine(base_dir=MODEL_DIR)
    yield


app = FastAPI(
    title="nanoGentzen-v3 Neuro-Symbolic Jev Server",
    description="Domain-agnostic formal verification and decision server with Saul Kripke semantics, "
                "Intuitionistic Relevance Logic, and Curry-Howard Proof Terms.",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# 5. API ENDPOINTS & PYTHON FUNCTIONS
# ============================================================================

@app.get("/health")
def health():
    if ENGINE is None:
        raise HTTPException(503, "Engine not initialized.")
    return {
        "status": "healthy",
        "engine": "nanoGentzen-v3",
        "device": ENGINE.device,
        "max_antecedents": getattr(ENGINE.model.config, "max_antecedents", 32),
        "num_epistemic_classes": getattr(ENGINE.model.config, "num_epistemic_classes", 3),
        "pointer_attention": getattr(ENGINE.model.config, "use_pointer_pivot", True),
    }


@app.post("/v1/systemone")
def systemone_api(req: DecideRequest):
    global ENGINE
    if ENGINE is None:
        ENGINE = ProductionGentzenEngine(base_dir=MODEL_DIR)
    return ENGINE.system_one(req)


@app.post("/v1/decide")
def decide_api(req: DecideRequest):
    global ENGINE
    if ENGINE is None:
        ENGINE = ProductionGentzenEngine(base_dir=MODEL_DIR)
    return ENGINE.decide(req)


def get_engine(device: Optional[str] = None) -> ProductionGentzenEngine:
    """Returns or initializes the singleton ProductionGentzenEngine."""
    global ENGINE
    if ENGINE is None:
        ENGINE = ProductionGentzenEngine(base_dir=MODEL_DIR, device=device)
    return ENGINE


def decide(req: Union[DecideRequest, Dict[str, Any]]) -> Dict[str, Any]:
    """In-process Python interface for benchmark evaluation and unit tests."""
    return get_engine().decide(req)


def systemone(req: Union[DecideRequest, Dict[str, Any]]) -> Dict[str, Any]:
    """In-process Python interface for System-1 benchmark evaluation."""
    return get_engine().system_one(req)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port)
