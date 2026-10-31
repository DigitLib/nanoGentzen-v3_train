"""
nanogentzen: Saul Kripke (1965) Epistemic Neural Prover with Span Pointer Sequent Attention and Dyckhoff G4ip Calculus.
"""
import os
import sys

# Ensure package directory and root directory are in path
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_PKG_DIR)
for _d in [_PKG_DIR, _ROOT_DIR]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from nanogentzen.kernel import (
    Formula,
    Var,
    Not,
    And,
    Or,
    Imp,
    Sequent,
    RULES,
    apply_rule,
    verify_proof_tree,
    verify_proof_tree_strict,
    canonicalize_sequent,
    restore_proof_tree,
    extract_kripke_countermodel,
    KripkeModel,
    KripkeWorld,
    extract_relevance_witnesses,
    proof_tree_to_curry_howard,
)

# Friendly alias
serialize_curry_howard = proof_tree_to_curry_howard

from nanogentzen.configuration_nanogentzen import GentzenConfig, PolicyValueConfig
from nanogentzen.tokenization_nanogentzen import LogicTokenizer, LogicTokenizerHF
from nanogentzen.modeling_nanogentzen import (
    GentzenPolicyValueModel,
    BidirectionalBlock,
)

GentzenPolicyValueNet = GentzenPolicyValueModel
from nanogentzen.domain_registry import (
    DomainCompilerRegistry,
    REGISTRY,
    register_domain,
)
from nanogentzen.parser import (
    parse_symbolic_sequent,
    FormulaParser,
    clean_term,
    compile_to_formula,
    ground_state_generic,
    detect_workflow,
    compile_structured_decision,
    get_calibrated_distribution,
)
from nanogentzen.dataset import (
    GentzenDataset,
    exhaustive_solver,
    synthesize_refuted_sequent,
    synthesize_horn_policy_trace,
    generate_random_formula,
    VARS_POOL,
)
from nanogentzen.search import NeuralProofSearch, prove_best_first

__version__ = "3.0.0"
__all__ = [
    # Logic Kernel
    "Formula",
    "Var",
    "Not",
    "And",
    "Or",
    "Imp",
    "Sequent",
    "RULES",
    "apply_rule",
    "verify_proof_tree",
    "verify_proof_tree_strict",
    "canonicalize_sequent",
    "restore_proof_tree",
    "extract_kripke_countermodel",
    "KripkeModel",
    "KripkeWorld",
    "extract_relevance_witnesses",
    "serialize_curry_howard",
    # Config & Tokenization
    "GentzenConfig",
    "PolicyValueConfig",
    "LogicTokenizer",
    "LogicTokenizerHF",
    # Neural Modeling
    "GentzenPolicyValueModel",
    "GentzenPolicyValueNet",
    "BidirectionalBlock",
    # Proof Search
    "NeuralProofSearch",
    "prove_best_first",
    # Parser & Compiler
    "DomainCompilerRegistry",
    "REGISTRY",
    "register_domain",
    "parse_symbolic_sequent",
    "FormulaParser",
    "clean_term",
    "compile_to_formula",
    "ground_state_generic",
    "detect_workflow",
    "compile_structured_decision",
    "get_calibrated_distribution",
    # Dataset & Synthesis
    "GentzenDataset",
    "exhaustive_solver",
    "synthesize_refuted_sequent",
    "synthesize_horn_policy_trace",
    "generate_random_formula",
    "VARS_POOL",
]
