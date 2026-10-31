"""
example_usage.py - Real-World Usage of nanoGentzen-v3 (HuggingFace + JEV Server APIs)
Demonstrates:
  1. `AutoModel.from_pretrained(..., trust_remote_code=True)` direct Python API
  2. System-2 Verified Decision (`model.decide` / `POST /v1/decide`) with Curry-Howard lambda-terms
  3. System-1 Ultra-Fast Epistemic Valuation (`model.system_one` / `POST /v1/systemone`) in ~2ms
  4. Saul Kripke (1965) 3-class epistemic countermodels for underdetermined/missing facts
"""

import json
import os
import torch
from transformers import AutoModel, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
bundle_dir = os.path.dirname(os.path.abspath(__file__))

print(f"[*] Loading nanoGentzen-v3 on {device.upper()} from: {bundle_dir}")
tokenizer = AutoTokenizer.from_pretrained(bundle_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(bundle_dir, trust_remote_code=True).to(device).eval()
print(f"[+] Model loaded successfully ({sum(p.numel() for p in model.parameters()):,} parameters).")

# ============================================================================
# Scenario 1: Enterprise Loan Approval (noul + choice + score)
# ============================================================================
compliance_payload = {
    "model": "nanoGentzen-v3",
    "state": {
        "applicant": {"income": 95000, "credit_score": 740, "employment_years": 4},
        "loan": {"amount": 250000, "purpose": "mortgage"},
    },
    "predicates": {
        "high_income": {"field": "applicant.income", "op": ">=", "value": 70000},
        "good_credit": {"field": "applicant.credit_score", "op": ">=", "value": 700},
        "stable_job": {"field": "applicant.employment_years", "op": ">=", "value": 2},
    },
    "policy_rules": [
        "high_income & good_credit => creditworthy",
        "creditworthy & stable_job => approve_loan",
        "approve_loan => tier_premium",
    ],
    "questions": {
        "approve_loan": {
            "type": "noul",
            "target": "approve_loan",
            "require_relevance": True,
        },
        "routing_action": {
            "type": "choice",
            "options": ["tier_premium", "tier_standard", "tier_reject"],
            "rules": [
                "tier_premium => tier_premium",
            ],
        },
        "risk_tier": {
            "type": "score",
            "levels": [0, 1, 2, 3],
            "level_atoms": ["risk_0", "risk_1", "risk_2", "risk_3"],
        },
    },
    "include_proof_tree": True,
    "include_lambda_term": True,
    "include_relevance": True,
}

print("\n" + "=" * 75)
print("1A. System-1 Ultra-Fast Epistemic Valuation (`model.system_one` / `/v1/systemone`)")
print("=" * 75)
res_s1 = model.system_one(compliance_payload)
print(f"Latency: {res_s1['latency_ms']} ms")
print(json.dumps(res_s1, indent=2))

print("\n" + "=" * 75)
print("1B. System-2 Verified Formal Decision (`model.decide` / `/v1/decide`)")
print("=" * 75)
res_s2 = model.decide(compliance_payload)
print(f"Latency: {res_s2['latency_ms']} ms")
for q_id, ans in res_s2["answers"].items():
    print(f"\nQuestion [{q_id}]:")
    print(f"  Type            : {ans.get('type')}")
    print(f"  Verification    : {ans.get('verification') or ans.get('verified')}")
    print(f"  Confidence      : {ans.get('confidence')}")
    print(f"  Probabilities   : {ans.get('probabilities')}")
    if "curry_howard_term" in ans:
        print(f"  Curry-Howard    : {ans['curry_howard_term']}")
    if "relevance" in ans:
        rel = ans["relevance"]
        print(f"  Relevant Witn.  : {rel.get('relevant_witnesses')}")
        print(f"  Strict Relevance: {rel.get('is_strictly_relevant')}")

# ============================================================================
# Scenario 2: Incomplete Evidence & Saul Kripke (1965) Countermodel
# ============================================================================
print("\n" + "=" * 75)
print("2. Incomplete Evidence Diagnosis (Saul Kripke 1965 Epistemic Countermodel)")
print("=" * 75)
missing_fact_payload = {
    "model": "nanoGentzen-v3",
    "state": {"applicant": {"credit_score": 740}},  # `income` and `employment_years` missing
    "predicates": {
        "high_income": {"field": "applicant.income", "op": ">=", "value": 70000},
        "good_credit": {"field": "applicant.credit_score", "op": ">=", "value": 700},
    },
    "policy_rules": ["high_income & good_credit => creditworthy"],
    "questions": {
        "creditworthy": {"type": "noul", "target": "creditworthy"},
    },
    "include_countermodel": True,
}

res_missing = model.decide(missing_fact_payload)
ans_m = res_missing["answers"]["creditworthy"]
print(f"Verification    : {ans_m['verification']}")
print(f"Epistemic Dist  : {ans_m['epistemic']}")
if "countermodel" in ans_m:
    cm = ans_m["countermodel"]
    print(f"Kripke Diagnosis: {cm.get('diagnosis')}")
    print(f"World G (known) : {cm.get('world_g_known_facts')}")
    print(f"World H (future): {cm.get('world_h_accessible_facts')}")

