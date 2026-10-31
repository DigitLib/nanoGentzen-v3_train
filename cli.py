#!/usr/bin/env python3
"""
cli.py - Command-Line Interface for nanoGentzen-v3 (`/v1/decide` & `/v1/systemone`)
Supports:
  1. File-based execution: `python cli.py -f request.json --mode decide`
  2. Quick queries: `python cli.py --state '{"age": 25}' --goal is_adult --rules 'age >= 18 => is_adult'`
  3. Interactive REPL shell: `python cli.py --interactive`
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from jev_server import ProductionGentzenEngine, DecideRequest, get_engine


def build_quick_payload(
    state_str: str,
    target: str,
    q_type: str = "noul",
    options: Optional[List[str]] = None,
    rules: Optional[List[str]] = None,
) -> dict:
    try:
        parsed_state = json.loads(state_str)
    except Exception:
        parsed_state = state_str

    q_spec: dict = {"type": q_type}
    if q_type == "noul":
        q_spec["target"] = target
    elif q_type == "choice":
        q_spec["options"] = options or [o.strip() for o in target.split(",") if o.strip()]
    elif q_type == "score":
        q_spec["levels"] = [0, 1, 2, 3]

    return {
        "model": "nanoGentzen-v3",
        "state": parsed_state,
        "policy_rules": rules or [],
        "questions": {target: q_spec},
    }


def interactive_shell(engine: ProductionGentzenEngine, mode: str = "decide"):
    print("\n" + "=" * 75)
    print(f" nanoGentzen-v3 Interactive Decision Shell (mode: {mode})")
    print(" Commands: ':mode decide', ':mode systemone', 'exit'")
    print(" Format  : <state / premises> ? <target goal>   OR raw JSON payload")
    print("=" * 75 + "\n")

    current_mode = mode
    while True:
        try:
            line = input(f"nanogentzen ({current_mode})> ").strip()
            if not line:
                continue
            if line.lower() in ("exit", "quit", "q"):
                break
            if line.startswith(":mode"):
                parts = line.split()
                if len(parts) > 1 and parts[1] in ("decide", "systemone"):
                    current_mode = parts[1]
                    print(f"[*] Switched mode to: {current_mode}")
                continue

            if line.startswith("{"):
                payload = json.loads(line)
            elif "?" in line:
                state_part, target_part = line.rsplit("?", 1)
                payload = build_quick_payload(
                    state_part.strip(), target_part.strip() or "target"
                )
            elif "|-" in line:
                state_part, target_part = line.split("|-", 1)
                premises = [p.strip() for p in state_part.split(",") if p.strip()]
                facts = [p for p in premises if "=>" not in p and "->" not in p]
                rules = [p for p in premises if "=>" in p or "->" in p]
                payload = {
                    "model": "nanoGentzen-v3",
                    "state": {f: True for f in facts},
                    "policy_rules": rules,
                    "questions": {target_part.strip(): {"type": "noul", "target": target_part.strip()}},
                }
            else:
                payload = build_quick_payload(line, "decision")

            req = DecideRequest(**payload)
            if current_mode == "decide":
                out = engine.decide(req)
            else:
                out = engine.system_one(req)
            print(json.dumps(out, indent=2))
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="nanoGentzen-v3 CLI Decision Runner")
    parser.add_argument("-f", "--file", type=str, help="Path to JSON payload file")
    parser.add_argument("-m", "--mode", choices=["decide", "systemone"], default="decide")
    parser.add_argument("-i", "--interactive", action="store_true", help="Launch interactive shell")
    parser.add_argument("--device", type=str, default=None, help="Execution device: 'cpu' or 'cuda'")
    parser.add_argument("--state", type=str, help="State JSON string or literal text")
    parser.add_argument("--goal", type=str, help="Goal atom for quick decision")
    parser.add_argument("--rules", nargs="*", default=[], help="Policy rules (e.g. 'P & Q => R')")

    args = parser.parse_args()
    engine = get_engine(device=args.device)

    if args.interactive:
        interactive_shell(engine, mode=args.mode)
        return

    if args.file:
        with open(args.file, "r") as f:
            payload = json.load(f)
        req = DecideRequest(**payload)
        out = engine.decide(req) if args.mode == "decide" else engine.system_one(req)
        print(json.dumps(out, indent=2))
        return

    if args.state and args.goal:
        payload = build_quick_payload(args.state, args.goal, rules=args.rules)
        req = DecideRequest(**payload)
        out = engine.decide(req) if args.mode == "decide" else engine.system_one(req)
        print(json.dumps(out, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()

