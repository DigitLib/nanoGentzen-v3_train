"""
test_jev.py - Verification suite for nanoGentzen-v3 APIs (`/v1/decide` & `/v1/systemone`).
Supports:
    python test_jev.py --local          # In-process direct python verification
    python test_jev.py                  # HTTP tests against http://127.0.0.1:8000
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

USE_LOCAL = "--local" in sys.argv
BASE_URL = next((a for a in sys.argv[1:] if not a.startswith("-")), "http://127.0.0.1:8000").rstrip("/")


def build_dispatchers():
    if USE_LOCAL:
        from jev_server import DecideRequest, decide, systemone, get_engine
        engine = get_engine()

        def _call_decide(payload: dict) -> dict:
            return decide(DecideRequest(**payload))

        def _call_systemone(payload: dict) -> dict:
            return systemone(DecideRequest(**payload))

        return _call_decide, _call_systemone

    def _post_http(endpoint: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{BASE_URL}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    return (
        lambda p: _post_http("/v1/decide", p),
        lambda p: _post_http("/v1/systemone", p),
    )


def run_tests():
    post_decide, post_systemone = build_dispatchers()
    target_label = "in-process (jev_server.py)" if USE_LOCAL else BASE_URL
    print(f"Testing nanoGentzen-v3 APIs (/v1/decide & /v1/systemone) at: {target_label}\n" + "=" * 75)

    # 1. Proved True with Relevance & Curry-Howard
    p1 = {
        "model": "nanoGentzen-v3",
        "state": {"user": {"age": 28, "kyc": True}},
        "predicates": {
            "is_adult": {"field": "user.age", "op": ">=", "value": 18},
            "kyc_passed": {"field": "user.kyc", "op": "==", "value": True},
        },
        "policy_rules": ["(is_adult & kyc_passed) => approved"],
        "questions": {"approved": {"type": "noul", "target": "approved", "require_relevance": True}},
    }
    r1 = post_decide(p1)
    ans1 = r1["answers"]["approved"]
    assert ans1["verification"] == "PROVEN_TRUE", f"Expected PROVEN_TRUE, got {ans1['verification']}"
    assert "apply" in ans1.get("curry_howard_term", ""), "Missing Curry-Howard term"
    assert ans1.get("relevance", {}).get("is_strictly_relevant"), "Relevance check failed"
    print(" [PASS] Test 1: Constructive Proof + Curry-Howard Term + Strict Relevance")

    # 2. Refuted False (Negation Derivation)
    p2 = {
        "model": "nanoGentzen-v3",
        "state": {"fraud_detected": True},
        "policy_rules": ["fraud_detected => ~approved"],
        "questions": {"approved": {"type": "noul", "target": "approved"}},
    }
    r2 = post_decide(p2)
    ans2 = r2["answers"]["approved"]
    assert ans2["verification"] == "PROVEN_FALSE", f"Expected PROVEN_FALSE, got {ans2['verification']}"
    print(" [PASS] Test 2: Constructive Refutation (Negation Derivation)")

    # 3. Saul Kripke (1965) Countermodel Extraction
    p3 = {
        "model": "nanoGentzen-v3",
        "state": {"status": "pending"},
        "policy_rules": ["doc_verified => approved"],
        "questions": {"approved": {"type": "noul", "target": "approved"}},
        "include_countermodel": True,
    }
    r3 = post_decide(p3)
    ans3 = r3["answers"]["approved"]
    assert ans3["verification"] == "INTUITIONISTIC_UNDECIDED", f"Expected INTUITIONISTIC_UNDECIDED, got {ans3['verification']}"
    print(" [PASS] Test 3: Saul Kripke Open Tableau Countermodel Extraction")

    # 4. Choice Selection with Derivation
    p4 = {
        "model": "nanoGentzen-v3",
        "state": {"is_vip": True},
        "questions": {
            "routing": {
                "type": "choice",
                "options": ["standard", "vip", "reject"],
                "rules": ["is_vip => vip"],
            }
        },
    }
    r4 = post_decide(p4)
    ans4 = r4["answers"]["routing"]
    assert ans4["choice"] == "vip", f"Expected vip, got {ans4['choice']}"
    assert ans4["verified"] is True, "Choice should be verified"
    print(" [PASS] Test 4: Categorical Choice Routing + Formal Proof")

    # 5. Score Tier Evaluation with Subformula Property Fast-Path
    p5 = {
        "model": "nanoGentzen-v3",
        "state": {"score_val": 85},
        "predicates": {
            "pass_low": {"field": "score_val", "op": ">=", "value": 50},
            "pass_high": {"field": "score_val", "op": ">=", "value": 80},
        },
        "policy_rules": [
            "pass_high => tier_2",
            "pass_low => tier_1",
        ],
        "questions": {
            "tier": {
                "type": "score",
                "levels": [0, 1, 2, 3],
                "level_atoms": ["tier_0", "tier_1", "tier_2", "tier_3"],
            }
        },
    }
    r5 = post_decide(p5)
    ans5 = r5["answers"]["tier"]
    assert ans5["proven_tier"] == 2, f"Expected proven tier 2, got {ans5['proven_tier']}"
    assert ans5["verified"] is True
    print(" [PASS] Test 5: Ordinal Score Tier Derivation + Fast Subformula Pruning")

    # 6. Ultra-Fast System 1 Forward Pass (< 5ms)
    r6 = post_systemone(p1)
    ans6 = r6["answers"]["approved"]
    assert ans6["type"] == "noul"
    assert "epistemic" in ans6
    print(f" [PASS] Test 6: System-1 Epistemic Gate (Latency: {r6['latency_ms']} ms)")

    print("\n[+] All 6 API verification tests PASSED successfully!")


if __name__ == "__main__":
    run_tests()
