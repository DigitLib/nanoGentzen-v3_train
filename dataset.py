"""
dataset.py (nanoGentzen-v3 Kripke + Jev)
Stage-3 Synthetic Corpus Generator for Gentzen Proof Trees, Multi-Hop Horn Policies
with Span Pointer Pivots (0..31), and 3-Class Saul Kripke Epistemic Supervision:
  - Class 2: PROVEN_TRUE (1.0)
  - Class 1: INTUITIONISTIC_UNDECIDED (0.5)
  - Class 0: PROVEN_FALSE (0.0)
"""
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset

try:
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
        extract_vars_set,
    )
    from nanogentzen.tokenization_nanogentzen import LogicTokenizer
except ImportError:
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
            extract_vars_set,
        )
        from tokenization_nanogentzen import LogicTokenizer
    except ImportError:
        from .kernel import (
            And,
            Formula,
            Imp,
            Not,
            Or,
            RULES,
            Sequent,
            Var,
            apply_rule,
            extract_vars_set,
        )
        from .tokenization_nanogentzen import LogicTokenizer


# 32 canonical variables supporting full 32-slot Span Pointer antecedent indexing
VARS_POOL: List[Var] = [Var("P"), Var("Q"), Var("R"), Var("S")] + [
    Var(f"v{i}") for i in range(28)
]


def generate_random_formula(depth: int = 2, vars_subset: Optional[List[Var]] = None) -> Formula:
    pool = vars_subset if vars_subset else VARS_POOL[:4]
    if depth <= 0 or random.random() < 0.25:
        return random.choice(pool)
    op = random.choice(["NOT", "AND", "OR", "IMP"])
    if op == "NOT":
        return Not(generate_random_formula(depth - 1, pool))
    if op == "AND":
        return And(
            generate_random_formula(depth - 1, pool),
            generate_random_formula(depth - 1, pool),
        )
    if op == "OR":
        return Or(
            generate_random_formula(depth - 1, pool),
            generate_random_formula(depth - 1, pool),
        )
    return Imp(
        generate_random_formula(depth - 1, pool),
        generate_random_formula(depth - 1, pool),
    )


def exhaustive_solver(
    seq: Sequent, depth: int = 0, max_depth: int = 6
) -> Optional[List[Tuple[Sequent, str, int]]]:
    """
    Deterministic backward solver aligned with Andreoli focusing (invertible phase first).
    Emits pivot = -100 for AXIOM and right-hand rules (R_*), and true antecedent index
    (0..31) strictly for left-hand rules (L_*).
    """
    if seq.is_axiom():
        return [(seq, "AXIOM", -100)]
    if depth >= max_depth:
        return None

    # 1. Single-premise invertible left rules (L_AND)
    for idx, target in enumerate(seq.gamma):
        if isinstance(target, And):
            premises = apply_rule(seq, "L_AND", idx=idx)
            if premises is not None:
                sub = []
                ok = True
                for p in premises:
                    sp = exhaustive_solver(p, depth + 1, max_depth)
                    if sp is None:
                        ok = False
                        break
                    sub.extend(sp)
                if ok:
                    return [(seq, "L_AND", idx)] + sub

    # 2. Invertible right decomposition rules (R_IMP, R_AND, R_NOT)
    for r in ["R_IMP", "R_AND", "R_NOT"]:
        premises = apply_rule(seq, r)
        if premises is not None:
            sub = []
            ok = True
            for p in premises:
                sp = exhaustive_solver(p, depth + 1, max_depth)
                if sp is None:
                    ok = False
                    break
                sub.extend(sp)
            if ok:
                return [(seq, r, -100)] + sub

    # 3. Two-premise invertible left rules (L_OR)
    for idx, target in enumerate(seq.gamma):
        if isinstance(target, Or):
            premises = apply_rule(seq, "L_OR", idx=idx)
            if premises is not None:
                sub = []
                ok = True
                for p in premises:
                    sp = exhaustive_solver(p, depth + 1, max_depth)
                    if sp is None:
                        ok = False
                        break
                    sub.extend(sp)
                if ok:
                    return [(seq, "L_OR", idx)] + sub

    # 4. Non-invertible right decomposition rules (R_OR_1, R_OR_2)
    for r in ["R_OR_1", "R_OR_2"]:
        premises = apply_rule(seq, r)
        if premises is not None:
            sub = []
            ok = True
            for p in premises:
                sp = exhaustive_solver(p, depth + 1, max_depth)
                if sp is None:
                    ok = False
                    break
                sub.extend(sp)
            if ok:
                return [(seq, r, -100)] + sub

    # 5. Non-invertible left decomposition rules across all pivots in Gamma (L_IMP, L_NOT)
    for idx in range(len(seq.gamma)):
        for r in ["L_IMP", "L_NOT"]:
            premises = apply_rule(seq, r, idx=idx)
            if premises is not None:
                sub = []
                ok = True
                for p in premises:
                    sp = exhaustive_solver(p, depth + 1, max_depth)
                    if sp is None:
                        ok = False
                        break
                    sub.extend(sp)
                if ok:
                    return [(seq, r, idx)] + sub

    # 6. Structural contraction fallback (L_CONTR) for implicational/negated premises
    if depth + 1 < max_depth:
        for idx, target in enumerate(seq.gamma):
            if isinstance(target, (Imp, Not)) and seq.gamma.count(target) < 2:
                premises = apply_rule(seq, "L_CONTR", idx=idx)
                if premises is not None:
                    sp = exhaustive_solver(premises[0], depth + 1, max_depth)
                    if sp is not None:
                        return [(seq, "L_CONTR", idx)] + sp

    return None


def synthesize_refuted_sequent() -> Sequent:
    """
    Generates a Class 0 (PROVEN_FALSE) sequent where Gamma constructively entails ~Delta
    or Gamma, Delta |- 0 holds, with shuffled antecedents and optional distractors
    to prevent positional index-0 shortcuts.
    """
    sample_vars = random.sample(VARS_POOL, 8)
    p, q, r, s = sample_vars[:4]
    distractor_pool = sample_vars[4:]

    patterns: List[Tuple[List[Formula], Formula]] = [
        # Direct 1-premise contradiction on goal: ~P |- P
        ([Not(p)], p),
        # Direct contradiction on goal with distractor: ~P, Q |- P
        ([Not(p), q], p),
        # Modus Tollens refutation: (P => Q), ~Q |- P
        ([Imp(p, q), Not(q)], p),
        # Negated conjunction refutation: ~P |- P & Q
        ([Not(p)], And(p, q)),
        # Disjunctive refutation: ~P, ~Q |- P | Q
        ([Not(p), Not(q)], Or(p, q)),
        # Policy violation chain: (P => ~Q), P |- Q
        ([Imp(p, Not(q)), p], q),
        # Multi-condition policy violation: ((P & Q) => ~R), P, Q |- R
        ([Imp(And(p, q), Not(r)), p, q], r),
        # Inconsistent antecedent premises: P, ~P |- Q
        ([p, Not(p)], q),
        # Chained contradiction: (P => Q), (Q => ~R), P |- R
        ([Imp(p, q), Imp(q, Not(r)), p], r),
        # Contrapositive chain refutation: (P => Q), (Q => R), ~R |- P
        ([Imp(p, q), Imp(q, r), Not(r)], p),
    ]
    core_gamma, goal = random.choice(patterns)
    gamma_list = list(core_gamma)

    # Optionally mix in 1..8 distractor facts/rules (leaving 50% clean without distractors)
    if random.random() < 0.50:
        num_extra = random.randint(1, 8)
        for j in range(num_extra):
            dv = distractor_pool[j % len(distractor_pool)]
            if j % 3 == 0 and len(distractor_pool) >= 2:
                dv2 = distractor_pool[(j + 1) % len(distractor_pool)]
                gamma_list.append(Imp(dv, dv2))
            else:
                gamma_list.append(dv)

    random.shuffle(gamma_list)
    return Sequent(tuple(gamma_list[:32]), (goal,))


def synthesize_constructive_theorem_trace() -> Tuple[Sequent, List[Tuple[Sequent, str, int]], str]:
    """
    Synthesizes structured constructive theorems and negative-goal refutation proofs
    that exercise all 11 Gentzen rules (including L_AND, L_OR, L_NOT, L_CONTR, R_IMP,
    R_AND, R_OR_1, R_OR_2, R_NOT) across shuffled antecedent pivot indices (0..31).
    """
    sample_vars = random.sample(VARS_POOL, k=min(20, len(VARS_POOL)))
    p, q, r, s = sample_vars[:4]
    d_vars = sample_vars[4:]

    # Build 0..14 atomic/implicational distractors (with 45% clean 0-distractor canonical theorems)
    num_dist = 0 if random.random() < 0.45 else random.randint(1, 14)
    distractors: List[Formula] = []
    for i in range(num_dist):
        dv = d_vars[i % len(d_vars)]
        if i % 4 == 0 and len(d_vars) >= 2:
            dv2 = d_vars[(i + 1) % len(d_vars)]
            distractors.append(Imp(dv, dv2))
        else:
            distractors.append(dv)

    kind = random.choice([
        "double_neg_lem_contr",
        "de_morgan_1",
        "de_morgan_2",
        "disjunctive_cases",
        "conjunctive_unpack",
        "neg_goal_refutation",
        "implication_transitivity",
        "or_intro_left_right",
    ])

    if kind == "double_neg_lem_contr":
        # distractors, |- ~(~(P | ~P)) : exercises R_NOT -> L_CONTR -> L_NOT -> R_OR_2 -> R_NOT -> L_NOT -> R_OR_1 -> AXIOM
        lem = Or(p, Not(p))
        n_lem = Not(lem)
        nn_lem = Not(n_lem)
        base_dist = tuple(distractors[:6])
        s0 = Sequent(base_dist, (nn_lem,))
        # Insert n_lem at a random position in gamma by shuffling s1's gamma or applying R_NOT then L_CONTR
        s1 = Sequent((n_lem,) + base_dist, ())
        s2 = Sequent((n_lem, n_lem) + base_dist, ())
        s3 = Sequent((n_lem,) + base_dist, (lem,))
        s4 = Sequent((n_lem,) + base_dist, (Not(p),))
        s5 = Sequent((p, n_lem) + base_dist, ())
        s6 = Sequent((p,) + base_dist, (lem,))
        s7 = Sequent((p,) + base_dist, (p,))
        trace = [
            (s0, "R_NOT", -100),
            (s1, "L_CONTR", 0),
            (s2, "L_NOT", 0),
            (s3, "R_OR_2", -100),
            (s4, "R_NOT", -100),
            (s5, "L_NOT", 1),
            (s6, "R_OR_1", -100),
            (s7, "AXIOM", -100),
        ]
        # Also include a shuffled variant where n_lem is at pivot k in 0..len(base_dist)
        if base_dist:
            g1_list = list(base_dist) + [n_lem]
            random.shuffle(g1_list)
            piv_k = g1_list.index(n_lem)
            s1_shuf = Sequent(tuple(g1_list), ())
            trace.insert(1, (s1_shuf, "L_CONTR", piv_k))
        return s0, trace, kind

    if kind == "de_morgan_1":
        # ~(P | Q), distractors |- (~P & ~Q)
        gamma = list(distractors[:6]) + [Not(Or(p, q))]
        random.shuffle(gamma)
        seq = Sequent(tuple(gamma), (And(Not(p), Not(q)),))
    elif kind == "de_morgan_2":
        # (~P & ~Q), distractors |- ~(P | Q)
        gamma = list(distractors[:6]) + [And(Not(p), Not(q))]
        random.shuffle(gamma)
        seq = Sequent(tuple(gamma), (Not(Or(p, q)),))
    elif kind == "disjunctive_cases":
        # (P | Q), (P => R), (Q => R), distractors |- R
        gamma = list(distractors[:8]) + [Or(p, q), Imp(p, r), Imp(q, r)]
        random.shuffle(gamma)
        seq = Sequent(tuple(gamma), (r,))
    elif kind == "conjunctive_unpack":
        # (P & Q), (P => R), (Q => S), distractors |- (R & S)
        gamma = list(distractors[:8]) + [And(p, q), Imp(p, r), Imp(q, s)]
        random.shuffle(gamma)
        seq = Sequent(tuple(gamma), (And(r, s),))
    elif kind == "neg_goal_refutation":
        # P, (P => ~Q), distractors |- ~Q  (System 2 constructive refutation trace)
        gamma = list(distractors[:10]) + [p, Imp(p, Not(q))]
        random.shuffle(gamma)
        seq = Sequent(tuple(gamma), (Not(q),))
    elif kind == "implication_transitivity":
        # (P => Q), (Q => R), distractors |- (P => R)
        gamma = list(distractors[:8]) + [Imp(p, q), Imp(q, r)]
        random.shuffle(gamma)
        seq = Sequent(tuple(gamma), (Imp(p, r),))
    else:
        # P, distractors |- (P | Q) or Q, distractors |- (P | Q)
        chosen = p if random.random() < 0.5 else q
        gamma = list(distractors[:10]) + [chosen]
        random.shuffle(gamma)
        seq = Sequent(tuple(gamma), (Or(p, q),))

    trace = exhaustive_solver(seq, max_depth=7)
    if trace:
        return seq, trace, kind

    fallback_seq = Sequent((p, Imp(p, q)), (q,))
    return fallback_seq, [(fallback_seq, "L_IMP", 1), (Sequent((p,), (p,)), "AXIOM", -100), (Sequent((q, p), (q,)), "AXIOM", -100)], "fallback"


def synthesize_horn_policy_trace() -> Tuple[Sequent, List[Tuple[Sequent, str, int]], Sequent, Sequent]:
    """
    Synthesizes a realistic multi-hop Horn-clause / enterprise policy sequent
    with 4 to 32 shuffled premises (facts + chained implications + distractors)
    exercising Span Pointer pivots across all 32 slots (0..31).

    Returns:
      - provable_root: Constructively provable sequent (Class 2, target_epistemic = 2, target_value = 1.0)
      - trace: Certified step-by-step derivation transitions [(seq, rule, pivot), ...]
      - undecided_seq: Same policy with 1 required premise removed (Class 1, target_epistemic = 1, target_value = 0.5)
      - refuted_seq: Same policy with a shuffled refutation/contradiction (Class 0, target_epistemic = 0, target_value = 0.0)
    """
    pool = random.sample(VARS_POOL, k=min(28, len(VARS_POOL)))
    hops = random.choice([1, 2, 2, 3])
    chain_vars = pool[: hops + 1]
    distractor_vars = pool[hops + 1 :]

    # Implication chain: c0 -> c1 -> ... -> c_hops (Goal)
    rules: List[Formula] = []
    facts: List[Formula] = [chain_vars[0]]
    extra_fact_var = distractor_vars[0] if distractor_vars else Var("v11")

    for i in range(hops):
        if i == 0 and random.random() < 0.45 and len(distractor_vars) >= 1:
            facts.append(extra_fact_var)
            rules.append(Imp(And(chain_vars[0], extra_fact_var), chain_vars[1]))
        else:
            rules.append(Imp(chain_vars[i], chain_vars[i + 1]))

    goal = chain_vars[hops]

    # Add 0..30 distractor facts and rules so antecedent count spans 1..32 (pivots 0..31)
    distractors: List[Formula] = []
    max_extra = max(2, 32 - len(facts) - len(rules))
    r_draw = random.random()
    if r_draw < 0.20:
        num_distractors = 0
    elif r_draw < 0.45:
        num_distractors = max_extra
    elif r_draw < 0.75 and max_extra >= 18:
        num_distractors = random.randint(18, max_extra)
    else:
        num_distractors = random.randint(1, max_extra)
    d_pool = distractor_vars[1:] if len(distractor_vars) > 1 else [Var("v8"), Var("v9"), Var("v10")]

    for j in range(num_distractors):
        dv = d_pool[j % len(d_pool)]
        # Limit compound Imp distractors to at most 5 per sequent so token length stays well below 256
        use_imp = (j < 10 and j % 2 == 0)
        if use_imp and len(d_pool) >= 2:
            dv2 = d_pool[(j + 1) % len(d_pool)]
            if dv != goal and dv2 != goal:
                distractors.append(Imp(dv, dv2))
        else:
            if dv != goal:
                distractors.append(dv)

    gamma_list = list(facts + rules + distractors)[:32]
    random.shuffle(gamma_list)
    provable_root = Sequent(tuple(gamma_list), (goal,))

    # Certified backward reduction (using -100 pivot for AXIOM and R_* steps)
    trace: List[Tuple[Sequent, str, int]] = []
    curr_seq = provable_root

    valid_trace = True
    for step_k in range(hops - 1, -1, -1):
        target_concl = chain_vars[step_k + 1]
        rule_idx = None
        for idx, f in enumerate(curr_seq.gamma):
            if isinstance(f, Imp) and f.right == target_concl:
                rule_idx = idx
                break
        if rule_idx is None:
            valid_trace = False
            break

        imp_formula = curr_seq.gamma[rule_idx]
        premises = apply_rule(curr_seq, "L_IMP", idx=rule_idx)
        if premises is None or len(premises) != 2:
            valid_trace = False
            break

        trace.append((curr_seq, "L_IMP", rule_idx))
        left_branch, right_branch = premises[0], premises[1]
        if right_branch.is_axiom():
            trace.append((right_branch, "AXIOM", -100))

        if isinstance(imp_formula.left, And):
            trace.append((left_branch, "R_AND", -100))
            and_premises = apply_rule(left_branch, "R_AND")
            if and_premises:
                for ap in and_premises:
                    if ap.is_axiom():
                        trace.append((ap, "AXIOM", -100))
            curr_seq = left_branch
            break
        else:
            curr_seq = left_branch
            if curr_seq.is_axiom():
                trace.append((curr_seq, "AXIOM", -100))

    if not valid_trace or not trace:
        fallback = exhaustive_solver(provable_root, max_depth=6)
        trace = fallback if fallback else [(Sequent((goal,), (goal,)), "AXIOM", -100)]

    # Paired Class 1 (INTUITIONISTIC_UNDECIDED): remove a required base fact (c0 or extra_fact_var)
    drop_fact = extra_fact_var if (len(facts) > 1 and random.random() < 0.5) else chain_vars[0]
    undecided_gamma = [f for f in gamma_list if f != drop_fact]
    undecided_seq = Sequent(tuple(undecided_gamma), (goal,))

    # Paired Class 0 (PROVEN_FALSE): diverse policy refutations with shuffled positions
    ref_mode = random.random()
    if ref_mode < 0.45:
        # Positive-fact policy violation: replace final rule (... => goal) with (... => ~goal)
        refuted_gamma = []
        for f in gamma_list:
            if isinstance(f, Imp) and f.right == goal:
                refuted_gamma.append(Imp(f.left, Not(goal)))
            else:
                refuted_gamma.append(f)
    elif ref_mode < 0.75:
        # Negative premise chain: ~c0 and (~c0 => ~goal) shuffled into gamma
        refuted_gamma = [f for f in gamma_list if f != chain_vars[0]][:30]
        refuted_gamma.append(Imp(Not(chain_vars[0]), Not(goal)))
        refuted_gamma.append(Not(chain_vars[0]))
    else:
        # Explicit blocking rule: blocker_var and (blocker_var => ~goal)
        blocker = d_pool[0] if d_pool else Var("v15")
        refuted_gamma = [f for f in gamma_list if f != chain_vars[0]][:30]
        refuted_gamma.append(blocker)
        refuted_gamma.append(Imp(blocker, Not(goal)))

    random.shuffle(refuted_gamma)
    refuted_seq = Sequent(tuple(refuted_gamma[:32]), (goal,))

    return provable_root, trace, undecided_seq, refuted_seq


class GentzenDataset(Dataset):
    """
    3-Class Saul Kripke PyTorch Dataset for nanoGentzen-v3.
    """
    def __init__(
        self,
        tokenizer: LogicTokenizer,
        num_samples: int = 6000,
        max_len: int = 256,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples: List[Dict[str, torch.Tensor]] = []
        self._build_dataset(num_samples, seed=seed)

    def _encode_padded(self, seq_str: str) -> List[int]:
        enc = self.tokenizer.encode(seq_str)[: self.max_len]
        return enc + [self.tokenizer.pad_id] * (self.max_len - len(enc))

    def _build_dataset(self, num_samples: int, seed: int = 42):
        random.seed(seed)
        torch.manual_seed(seed)
        target_pos = max(1, num_samples // 3)
        target_undec = max(1, num_samples // 3)
        target_false = max(1, num_samples - target_pos - target_undec)

        class_2_count = 0
        class_1_count = 0
        class_0_count = 0
        trajectory_id = 0

        while len(self.samples) < num_samples:
            trajectory_id += 1

            if class_2_count < target_pos and len(self.samples) < num_samples:
                # Class 2: PROVEN_TRUE (Horn policy traces + structured theorems + random theorems)
                mode = random.random()
                if mode < 0.55:
                    _, trace, _, _ = synthesize_horn_policy_trace()
                elif mode < 0.80:
                    _, trace, _ = synthesize_constructive_theorem_trace()
                else:
                    ant_count = random.randint(0, 3)
                    vars_sub = random.sample(VARS_POOL, k=4)
                    gamma = tuple(
                        generate_random_formula(depth=random.randint(0, 2), vars_subset=vars_sub)
                        for _ in range(ant_count)
                    )
                    delta = (generate_random_formula(depth=random.randint(1, 2), vars_subset=vars_sub),)
                    trace = exhaustive_solver(Sequent(gamma, delta), max_depth=6)

                if trace:
                    for s, rule, pivot in trace:
                        if class_2_count >= target_pos or len(self.samples) >= num_samples:
                            break
                        is_left_rule = rule.startswith("L_")
                        p_clamped = min(max(pivot, 0), 31) if (is_left_rule and pivot >= 0) else -100
                        r_idx = RULES.index(rule) if rule in RULES else -100
                        self.samples.append(
                            {
                                "input_ids": torch.tensor(self._encode_padded(s.to_str()), dtype=torch.long),
                                "target_rule": torch.tensor(r_idx, dtype=torch.long),
                                "target_pivot": torch.tensor(p_clamped, dtype=torch.long),
                                "target_epistemic": torch.tensor(2, dtype=torch.long),
                                "target_value": torch.tensor(1.0, dtype=torch.float32),
                                "trajectory_id": torch.tensor(trajectory_id, dtype=torch.long),
                            }
                        )
                        class_2_count += 1

            if class_1_count < target_undec and len(self.samples) < num_samples:
                # Class 1: INTUITIONISTIC_UNDECIDED (Missing premises, fallacies, open countermodels)
                u_mode = random.random()
                if u_mode < 0.50:
                    _, _, undec_s, _ = synthesize_horn_policy_trace()
                elif u_mode < 0.80:
                    p, q = random.sample(VARS_POOL, 2)
                    fallacies = [
                        Sequent((), (Or(p, Not(p)),)),                           # LEM
                        Sequent((), (Imp(Imp(Imp(p, q), p), p),)),              # Peirce
                        Sequent((Not(Not(p)),), (p,)),                          # DNE
                        Sequent((Imp(p, q), q), (p,)),                          # Affirming Consequent
                        Sequent((Imp(p, q), Not(p)), (Not(q),)),                # Denying Antecedent
                        Sequent((p,), (q,)),                                    # Unrelated Var
                    ]
                    undec_s = random.choice(fallacies)
                else:
                    vars_sub = random.sample(VARS_POOL, k=3)
                    gamma = (generate_random_formula(depth=1, vars_subset=vars_sub),)
                    delta = (generate_random_formula(depth=random.randint(1, 2), vars_subset=vars_sub),)
                    cand_s = Sequent(gamma, delta)
                    if (
                        exhaustive_solver(cand_s, max_depth=5) is None
                        and exhaustive_solver(Sequent(gamma, (Not(delta[0]),)), max_depth=5) is None
                    ):
                        undec_s = cand_s
                    else:
                        _, _, undec_s, _ = synthesize_horn_policy_trace()

                self.samples.append(
                    {
                        "input_ids": torch.tensor(self._encode_padded(undec_s.to_str()), dtype=torch.long),
                        "target_rule": torch.tensor(-100, dtype=torch.long),
                        "target_pivot": torch.tensor(-100, dtype=torch.long),
                        "target_epistemic": torch.tensor(1, dtype=torch.long),
                        "target_value": torch.tensor(0.5, dtype=torch.float32),
                        "trajectory_id": torch.tensor(trajectory_id, dtype=torch.long),
                    }
                )
                class_1_count += 1

            if class_0_count < target_false and len(self.samples) < num_samples:
                # Class 0: PROVEN_FALSE (Shuffled policy refutations & constructive contradictions)
                if random.random() < 0.5:
                    _, _, _, refuted_s = synthesize_horn_policy_trace()
                else:
                    refuted_s = synthesize_refuted_sequent()
                self.samples.append(
                    {
                        "input_ids": torch.tensor(self._encode_padded(refuted_s.to_str()), dtype=torch.long),
                        "target_rule": torch.tensor(-100, dtype=torch.long),
                        "target_pivot": torch.tensor(-100, dtype=torch.long),
                        "target_epistemic": torch.tensor(0, dtype=torch.long),
                        "target_value": torch.tensor(0.0, dtype=torch.float32),
                        "trajectory_id": torch.tensor(trajectory_id, dtype=torch.long),
                    }
                )
                class_0_count += 1

            if (
                class_2_count >= target_pos
                and class_1_count >= target_undec
                and class_0_count >= target_false
                and len(self.samples) < num_samples
            ):
                target_false += (num_samples - len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        torch.save(self.samples, filepath)
        print(f"[+] Dataset saved to {filepath} ({len(self.samples)} samples)")

    @classmethod
    def load(
        cls,
        filepath: str,
        tokenizer: Optional[LogicTokenizer] = None,
        max_len: int = 256,
    ) -> "GentzenDataset":
        if tokenizer is None:
            tokenizer = LogicTokenizer()
        ds = cls.__new__(cls)
        super(GentzenDataset, ds).__init__()
        ds.tokenizer = tokenizer
        ds.max_len = max_len
        ds.samples = torch.load(filepath, map_location="cpu", weights_only=False)
        print(f"[+] Loaded {len(ds.samples)} samples from {filepath}")
        return ds
