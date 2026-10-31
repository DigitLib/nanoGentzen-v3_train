"""
test_pipeline.py (nanoGentzen-v3 Kripke + Jev)
Comprehensive test suite verifying the entire nanoGentzen-v3 training pipeline:
  1. Gentzen Logic Kernel & Strict Proof-Tree Verification
  2. Intuitionistic Relevance (IR) & Curry-Howard Lambda Serialization
  3. Saul Kripke Open Tableau Countermodel Extraction
  4. LogicTokenizer Encoding, Decoding, and Vocabulary
  5. 3-Class Balanced Dataset Generation & DataLoader
  6. GentzenPolicyValueModel Forward & Loss Calculation
  7. Gradient Flow & Optimizer Step
  8. Neural Proof Search Integration
"""
import os
import sys
import tempfile
import unittest

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from torch.utils.data import DataLoader

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
    verify_proof_tree_strict,
    extract_kripke_countermodel,
    extract_relevance_witnesses,
    proof_tree_to_curry_howard,
)
from nanogentzen.configuration_nanogentzen import GentzenConfig
from nanogentzen.tokenization_nanogentzen import LogicTokenizer
from nanogentzen.modeling_nanogentzen import GentzenPolicyValueModel
from nanogentzen.dataset import (
    GentzenDataset,
    synthesize_horn_policy_trace,
    synthesize_refuted_sequent,
    exhaustive_solver,
)
from nanogentzen.search import NeuralProofSearch


class TestNanoGentzenPipeline(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = LogicTokenizer()
        self.config = GentzenConfig(
            vocab_size=self.tokenizer.vocab_size,
            block_size=128,
            n_layer=2,
            n_head=4,
            n_embd=64,
            num_rules=11,
            max_antecedents=32,
            use_pointer_pivot=True,
            pointer_dim=32,
            num_epistemic_classes=3,
        )

    def test_01_kernel_deduction_and_verification(self):
        """Tests Gentzen deduction rules, axiom detection, and strict tree verification."""
        P, Q = Var("P"), Var("Q")
        seq = Sequent((P, Q), (And(P, Q),))
        premises = apply_rule(seq, "R_AND")
        self.assertIsNotNone(premises)
        self.assertEqual(len(premises), 2)
        self.assertTrue(premises[0].is_axiom())
        self.assertTrue(premises[1].is_axiom())

        proof_tree = {
            "sequent": seq.to_str(),
            "rule": "R_AND",
            "branches": [
                {"sequent": premises[0].to_str(), "rule": "AXIOM", "branches": []},
                {"sequent": premises[1].to_str(), "rule": "AXIOM", "branches": []},
            ],
        }
        self.assertTrue(verify_proof_tree_strict(proof_tree))

    def test_02_relevance_and_curry_howard(self):
        """Tests Intuitionistic Relevance witness tracking and Curry-Howard terms."""
        P, Q, R = Var("P"), Var("Q"), Var("R")
        root_seq = Sequent((P, Q, Imp(P, R)), (R,))
        proof_tree = {
            "sequent": root_seq.to_str(),
            "rule": "L_IMP_2",
            "branches": [
                {"sequent": "P, Q |- P", "rule": "AXIOM", "branches": []},
                {"sequent": "R, P, Q |- R", "rule": "AXIOM", "branches": []},
            ],
        }
        relevance = extract_relevance_witnesses(root_seq, proof_tree)
        self.assertIn("P", relevance["relevant_witnesses"])
        self.assertIn("(P => R)", relevance["relevant_witnesses"])
        self.assertIn("Q", relevance["unused_premises"])

        ch_term = proof_tree_to_curry_howard(proof_tree)
        self.assertTrue(len(ch_term) > 0)

    def test_03_kripke_countermodel_extraction(self):
        """Tests Kripke countermodel extraction on open/undecided and refuted sequents."""
        P, Q = Var("P"), Var("Q")
        # Missing fact policy: P, (P & Q => R) |- R
        R = Var("R")
        missing_seq = Sequent((P, Imp(And(P, Q), R)), (R,))
        cm = extract_kripke_countermodel(missing_seq)
        self.assertIsNotNone(cm)
        self.assertIn("Q", cm["missing_facts"])

        # Contradiction: ~P, P |- Q
        refuted_seq = Sequent((Not(P), P), (Q,))
        cm_ref = extract_kripke_countermodel(refuted_seq)
        self.assertIsNotNone(cm_ref)
        self.assertEqual(cm_ref.get("diagnosis"), "PROVEN_FALSE")

    def test_04_tokenizer_properties(self):
        """Tests tokenizer vocab, encoding, decoding, and special tokens."""
        seq_str = "P, (P => Q) |- Q"
        tokens = self.tokenizer.encode(seq_str)
        self.assertTrue(len(tokens) > 0)
        self.assertIn(self.tokenizer.turnstile_id, tokens)
        self.assertIsInstance(self.tokenizer.vocab, dict)
        self.assertGreater(len(self.tokenizer.vocab), 30)

    def test_05_dataset_synthesis(self):
        """Tests 3-class dataset generation and DataLoader batching."""
        dataset = GentzenDataset(self.tokenizer, num_samples=30, max_len=128, seed=42)
        self.assertEqual(len(dataset), 30)

        loader = DataLoader(dataset, batch_size=10, shuffle=False)
        batch = next(iter(loader))
        self.assertIn("input_ids", batch)
        self.assertIn("target_rule", batch)
        self.assertIn("target_pivot", batch)
        self.assertIn("target_epistemic", batch)
        self.assertEqual(batch["input_ids"].shape, (10, 128))

    def test_06_model_forward_and_loss(self):
        """Tests model forward pass, span pointer attention, and joint loss."""
        model = GentzenPolicyValueModel(self.config).to(self.device)
        model.eval()

        input_ids = torch.randint(0, self.config.vocab_size, (4, 64), device=self.device)
        target_rule = torch.tensor([0, 1, -100, 2], dtype=torch.long, device=self.device)
        target_pivot = torch.tensor([0, 2, -100, 1], dtype=torch.long, device=self.device)
        target_epistemic = torch.tensor([2, 1, 0, 2], dtype=torch.long, device=self.device)

        rule_logits, pivot_logits, val_pred, loss = model(
            input_ids=input_ids,
            targets_rule=target_rule,
            targets_pivot=target_pivot,
            targets_epistemic=target_epistemic,
        )
        self.assertEqual(rule_logits.shape, (4, self.config.num_rules))
        self.assertEqual(pivot_logits.shape, (4, self.config.max_antecedents))
        self.assertEqual(val_pred.shape, (4,))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))

    def test_07_backward_and_optimization_step(self):
        """Tests gradient backpropagation and parameter update."""
        model = GentzenPolicyValueModel(self.config).to(self.device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        input_ids = torch.randint(0, self.config.vocab_size, (4, 64), device=self.device)
        target_rule = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=self.device)
        target_pivot = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=self.device)
        target_epistemic = torch.tensor([2, 1, 0, 2], dtype=torch.long, device=self.device)

        init_param = next(model.parameters()).clone()

        optimizer.zero_grad()
        _, _, _, loss = model(
            input_ids=input_ids,
            targets_rule=target_rule,
            targets_pivot=target_pivot,
            targets_epistemic=target_epistemic,
        )
        loss.backward()
        optimizer.step()

        updated_param = next(model.parameters())
        self.assertFalse(torch.allclose(init_param, updated_param))

    def test_08_search_integration(self):
        """Tests NeuralProofSearch on an identity sequent."""
        model = GentzenPolicyValueModel(self.config).to(self.device)
        searcher = NeuralProofSearch(model, self.tokenizer, device=self.device)

        P = Var("P")
        seq = Sequent((), (Imp(P, P),))
        proof, countermodel = searcher.prove(seq, max_depth=4)
        self.assertIsNotNone(proof)
        self.assertTrue(verify_proof_tree_strict(proof))


if __name__ == "__main__":
    print(f"[*] Running nanoGentzen-v3 pipeline integration tests...")
    unittest.main()
