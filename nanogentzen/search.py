"""
search.py (nanoGentzen-v3 Kripke + Jev)
Neural Proof Search Engine with:
  1. Andreoli Asynchronous Invertible Reduction (Deterministic CPU Fast-Path)
  2. Span Pointer Synchronous Choice Selection
  3. Saul Kripke (1965) 3-Class Epistemic Pruning
  4. Open Semantic Tableau Countermodel Extraction on Proof Failure
"""
from typing import Any, Dict, List, Optional, Set, Tuple
import torch
import torch.nn.functional as F

try:
    from kernel import (
        And,
        Imp,
        Not,
        Or,
        RULES,
        Sequent,
        Var,
        apply_rule,
        canonicalize_sequent,
        extract_kripke_countermodel,
        restore_proof_tree,
        INVERTIBLE_LEFT_RULES,
        INVERTIBLE_RIGHT_RULES,
    )
except ImportError:
    try:
        from nanogentzen.kernel import (
            And,
            Imp,
            Not,
            Or,
            RULES,
            Sequent,
            Var,
            apply_rule,
            canonicalize_sequent,
            extract_kripke_countermodel,
            restore_proof_tree,
            INVERTIBLE_LEFT_RULES,
            INVERTIBLE_RIGHT_RULES,
        )
    except ImportError:
        from .kernel import (
            And,
            Imp,
            Not,
            Or,
            RULES,
            Sequent,
            Var,
            apply_rule,
            canonicalize_sequent,
            extract_kripke_countermodel,
            restore_proof_tree,
            INVERTIBLE_LEFT_RULES,
            INVERTIBLE_RIGHT_RULES,
        )

try:
    from modeling_nanogentzen import GentzenPolicyValueModel
except ImportError:
    try:
        from nanogentzen.model import GentzenPolicyValueModel
    except ImportError:
        from .modeling_nanogentzen import GentzenPolicyValueModel

try:
    from tokenization_nanogentzen import LogicTokenizer
except ImportError:
    try:
        from nanogentzen.tokenizer import LogicTokenizer
    except ImportError:
        from .tokenization_nanogentzen import LogicTokenizer


class NeuralProofSearch:
    def __init__(
        self,
        model: GentzenPolicyValueModel,
        tokenizer: LogicTokenizer,
        device: str = "cuda",
        value_prune_threshold: float = 0.05,
        use_alpha_renaming: bool = True,
        use_invertible_fastpath: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.value_prune_threshold = value_prune_threshold
        self.use_alpha_renaming = use_alpha_renaming
        self.use_invertible_fastpath = use_invertible_fastpath
        self.model.eval()

    def _find_invertible_action(self, seq: Sequent) -> Optional[Tuple[str, int]]:
        """
        Andreoli Asynchronous Phase:
        Deterministically returns the first applicable 100% invertible rule on CPU.
        Invertible rules preserve provability, eliminating choice points and neural overhead.
        """
        # 1. Single-premise left invertible rules
        for idx, target in enumerate(seq.gamma):
            t_name = getattr(target, "__class__", type(target)).__name__
            if (isinstance(target, And) or t_name == "And") and apply_rule(seq, "L_AND", idx) is not None:
                return ("L_AND", idx)

        # 2. Right invertible rules
        if seq.delta:
            d = seq.delta[0]
            d_name = getattr(d, "__class__", type(d)).__name__
            if (isinstance(d, Imp) or d_name == "Imp") and apply_rule(seq, "R_IMP", 0) is not None:
                return ("R_IMP", 0)
            if (isinstance(d, And) or d_name == "And") and apply_rule(seq, "R_AND", 0) is not None:
                return ("R_AND", 0)
            if (isinstance(d, Not) or d_name == "Not") and apply_rule(seq, "R_NOT", 0) is not None:
                return ("R_NOT", 0)

        # 3. Two-premise left invertible rules (L_OR)
        for idx, target in enumerate(seq.gamma):
            t_name = getattr(target, "__class__", type(target)).__name__
            if (isinstance(target, Or) or t_name == "Or") and apply_rule(seq, "L_OR", idx) is not None:
                return ("L_OR", idx)

        return None

    @torch.no_grad()
    def evaluate_sequent(
        self, seq: Sequent
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Forward pass returning (rule_probs, pivot_probs, epistemic_probs, constructive_value).
        """
        eval_seq = canonicalize_sequent(seq)[0] if self.use_alpha_renaming else seq
        encoded = self.tokenizer.encode(eval_seq.to_str())[: self.model.config.block_size]
        padded = encoded + [self.tokenizer.pad_id] * (self.model.config.block_size - len(encoded))
        x = torch.tensor([padded], dtype=torch.long, device=self.device)

        rule_logits, pivot_logits, constructive_val, _ = self.model(input_ids=x)
        rule_probs = F.softmax(rule_logits[0], dim=-1)
        pivot_probs = F.softmax(pivot_logits[0], dim=-1)
        epistemic_probs = self.model._last_epistemic_probs[0]
        val_score = float(constructive_val[0].item())

        return rule_probs, pivot_probs, epistemic_probs, val_score

    def evaluate_value(self, seq: Sequent) -> float:
        """Evaluates scalar epistemic value for a sequent."""
        return self.evaluate_sequent(seq)[3]

    @torch.no_grad()
    def rank_actions(
        self, seq: Sequent
    ) -> Tuple[List[Tuple[str, int]], torch.Tensor, float]:
        """
        Ranks applicable Gentzen rules and Span Pointer pivots by joint score:
        Score = P(Rule) * P(Pivot).
        """
        rule_probs, pivot_probs, ep_probs, val_score = self.evaluate_sequent(seq)
        delta = seq.delta[0] if seq.delta else None
        d_name = getattr(delta, "__class__", type(delta)).__name__ if delta else ""
        gamma = seq.gamma
        gamma_len = len(gamma)
        max_ant = getattr(self.model.config, "max_antecedents", 32)
        scored_actions: List[Tuple[float, str, int]] = []

        for r_idx, rule in enumerate(RULES):
            if r_idx >= rule_probs.size(-1):
                continue
            r_prob = float(rule_probs[r_idx].item())

            # Check structural preconditions
            if rule == "R_IMP" and (not delta or not (isinstance(delta, Imp) or d_name == "Imp")):
                continue
            if rule == "R_AND" and (not delta or not (isinstance(delta, And) or d_name == "And")):
                continue
            if rule in ("R_OR_1", "R_OR_2") and (not delta or not (isinstance(delta, Or) or d_name == "Or")):
                continue
            if rule == "R_NOT" and (not delta or not (isinstance(delta, Not) or d_name == "Not")):
                continue

            if rule.startswith("R_"):
                scored_actions.append((r_prob, rule, 0))
            elif rule == "L_CONTR":
                for p in range(min(gamma_len, max_ant)):
                    t_p = gamma[p]
                    tp_name = getattr(t_p, "__class__", type(t_p)).__name__
                    if (isinstance(t_p, (Imp, Not)) or tp_name in ("Imp", "Not")) and gamma.count(t_p) < 2:
                        p_prob = float(pivot_probs[p].item()) if p < pivot_probs.size(-1) else 0.001
                        scored_actions.append((r_prob * p_prob, rule, p))
            elif rule.startswith("L_"):
                for p in range(min(gamma_len, max_ant)):
                    f = gamma[p]
                    f_name = getattr(f, "__class__", type(f)).__name__
                    valid = False
                    if rule == "L_IMP" and (isinstance(f, Imp) or f_name == "Imp"):
                        valid = True
                    elif rule == "L_AND" and (isinstance(f, And) or f_name == "And"):
                        valid = True
                    elif rule == "L_OR" and (isinstance(f, Or) or f_name == "Or"):
                        valid = True
                    elif rule == "L_NOT" and (isinstance(f, Not) or f_name == "Not"):
                        valid = True

                    if valid:
                        p_prob = float(pivot_probs[p].item()) if p < pivot_probs.size(-1) else 0.001
                        scored_actions.append((r_prob * p_prob, rule, p))

        scored_actions.sort(key=lambda x: x[0], reverse=True)
        return [(r, idx) for _, r, idx in scored_actions], ep_probs, val_score

    def _prove_recursive(
        self,
        seq: Sequent,
        depth: int,
        max_depth: int,
        budget: List[int],
        max_nodes: int,
        path_visited: Set[str],
        memo: Dict[str, Optional[Dict]],
        truncated: Optional[List[bool]] = None,
    ) -> Optional[Dict]:
        if seq.is_axiom():
            return {"sequent": seq.to_str(), "rule": "AXIOM", "branches": []}

        if depth >= max_depth or budget[0] >= max_nodes:
            if truncated is not None:
                truncated[0] = True
            return None

        seq_key = seq.to_str()
        if seq_key in path_visited:
            if truncated is not None:
                truncated[0] = True
            return None
        if seq_key in memo:
            return memo[seq_key]

        path_visited.add(seq_key)
        budget[0] += 1
        sub_trunc = [False]

        # Phase 1: Andreoli Asynchronous Invertible Reduction
        if self.use_invertible_fastpath:
            inv_action = self._find_invertible_action(seq)
            if inv_action is not None:
                rule, idx = inv_action
                premises = apply_rule(seq, rule, idx=idx)
                if premises is not None:
                    sub_proofs = []
                    failed = False
                    for p in premises:
                        branch_trunc = [False]
                        res = self._prove_recursive(
                            p, depth + 1, max_depth, budget, max_nodes, path_visited, memo, branch_trunc
                        )
                        if branch_trunc[0]:
                            sub_trunc[0] = True
                        if res is None:
                            failed = True
                            break
                        sub_proofs.append(res)
                    path_visited.remove(seq_key)
                    if not failed and len(sub_proofs) == len(premises):
                        tree = {
                            "sequent": seq.to_str(),
                            "rule": f"{rule}_{idx}" if "L_" in rule else rule,
                            "branches": sub_proofs,
                        }
                        memo[seq_key] = tree
                        return tree
                    if not sub_trunc[0]:
                        memo[seq_key] = None
                    elif truncated is not None:
                        truncated[0] = True
                    return None

        # Phase 2: Synchronous Choice Points with Kripke Epistemic Guidance
        candidate_actions, ep_probs, val_score = self.rank_actions(seq)

        # Epistemic pruning: if P(False) is high or constructive value is near zero
        p_false = float(ep_probs[0].item()) if ep_probs is not None else 0.0
        if depth > 1 and (p_false > 0.90 or val_score < self.value_prune_threshold):
            path_visited.remove(seq_key)
            if truncated is not None:
                truncated[0] = True
            return None

        for rule, idx in candidate_actions:
            if budget[0] >= max_nodes:
                sub_trunc[0] = True
                break
            premises = apply_rule(seq, rule, idx=idx)
            if premises is not None:
                sub_proofs = []
                failed = False
                for p in premises:
                    branch_trunc = [False]
                    res = self._prove_recursive(
                        p, depth + 1, max_depth, budget, max_nodes, path_visited, memo, branch_trunc
                    )
                    if branch_trunc[0]:
                        sub_trunc[0] = True
                    if res is None:
                        failed = True
                        break
                    sub_proofs.append(res)

                if not failed and len(sub_proofs) == len(premises):
                    tree = {
                        "sequent": seq.to_str(),
                        "rule": f"{rule}_{idx}" if "L_" in rule else rule,
                        "branches": sub_proofs,
                    }
                    path_visited.remove(seq_key)
                    memo[seq_key] = tree
                    return tree

        path_visited.remove(seq_key)
        if not sub_trunc[0]:
            memo[seq_key] = None
        elif truncated is not None:
            truncated[0] = True
        return None

    def prove(
        self,
        seq: Sequent,
        max_depth: int = 8,
        max_nodes: int = 300,
        extract_countermodel_on_failure: bool = True,
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Top-level proof search entrypoint.
        Returns (proof_tree, countermodel_diagnosis).
        If proof succeeds, proof_tree is returned.
        If proof fails, an open Kripke tableau countermodel is extracted.
        """
        budget = [0]
        path_visited: Set[str] = set()
        memo: Dict[str, Optional[Dict]] = {}

        if self.use_alpha_renaming:
            canon_seq, _, rev_map = canonicalize_sequent(seq)
            raw_tree = self._prove_recursive(
                canon_seq, 0, max_depth, budget, max_nodes, path_visited, memo
            )
            proof_tree = restore_proof_tree(raw_tree, rev_map)
        else:
            proof_tree = self._prove_recursive(
                seq, 0, max_depth, budget, max_nodes, path_visited, memo
            )

        countermodel = None
        if proof_tree is None and extract_countermodel_on_failure:
            countermodel = extract_kripke_countermodel(seq, max_depth=max_depth)

        return proof_tree, countermodel


def prove_best_first(
    model: GentzenPolicyValueModel,
    tokenizer: LogicTokenizer,
    seq: Sequent,
    device: str = "cuda",
    max_depth: int = 8,
    max_nodes: int = 300,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Convenience wrapper for one-shot neural proof search with Kripke diagnosis."""
    searcher = NeuralProofSearch(model, tokenizer, device=device)
    return searcher.prove(seq, max_depth=max_depth, max_nodes=max_nodes)

