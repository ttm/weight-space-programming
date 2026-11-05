"""Command line interface: python -m wsp ..."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation import evaluate, random_baseline, stress
from .network import load_candidate
from .repair import load_patch, repair_report, apply_patch
from .session import finalize, probe, start
from .tasks import TASKS


def _round_floats(data: object, digits: int = 12) -> object:
    """Round floats to `digits` significant digits so reports are byte-identical across platforms.

    Random baselines and noise perturbations draw weights with random.gauss, which calls the C library's log/cos/sin;
    their last bit differs between operating systems and CPU architectures.
    """
    if isinstance(data, float):
        return float(f"{data:.{digits}g}")
    if isinstance(data, dict):
        return {key: _round_floats(value, digits) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_round_floats(value, digits) for value in data]
    return data


def _print(data: object, output: str | None = None) -> None:
    contents = json.dumps(_round_floats(data), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf8")
    print(contents, end="")


def main() -> None:
    parser = argparse.ArgumentParser(prog="wsp", description="Direct neural weight benchmark, no training")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    desc = sub.add_parser("describe"); desc.add_argument("task", choices=TASKS)
    ev = sub.add_parser("evaluate"); ev.add_argument("candidate"); ev.add_argument("--splits", nargs="+", default=["test", "ood", "edge"]); ev.add_argument("--output")
    bm = sub.add_parser("benchmark"); bm.add_argument("--directory", default="examples"); bm.add_argument("--output"); bm.add_argument("--random-trials", type=int, default=30)
    rp = sub.add_parser("repair"); rp.add_argument("candidate"); rp.add_argument("patch"); rp.add_argument("--output"); rp.add_argument("--candidate-output")
    st = sub.add_parser("stress"); st.add_argument("candidate"); st.add_argument("--output")
    se = sub.add_parser("start"); se.add_argument("task", choices=TASKS); se.add_argument("--session", required=True); se.add_argument("--mode", choices=["blind", "budgeted", "open"], default="blind"); se.add_argument("--explanation", choices=["required", "optional", "forbidden"], default="optional"); se.add_argument("--budget", type=int, default=3)
    se.add_argument("--participant", help="who or what writes the candidates"); se.add_argument("--model", help="model name and version"); se.add_argument("--prompt-file", help="prompt given to the participant; stored as a SHA-256 hash")
    pr = sub.add_parser("probe"); pr.add_argument("candidate"); pr.add_argument("--session", required=True)
    fi = sub.add_parser("finalize"); fi.add_argument("candidate"); fi.add_argument("--session", required=True)
    args = parser.parse_args()
    try:
        if args.command == "list":
            _print([vars(t) for t in TASKS.values()])
        elif args.command == "describe":
            _print(vars(TASKS[args.task]))
        elif args.command == "evaluate":
            _print(evaluate(load_candidate(args.candidate), args.splits), args.output)
        elif args.command == "repair":
            cand, patch = load_candidate(args.candidate), load_patch(args.patch)
            report = repair_report(cand, patch)
            if args.candidate_output:
                Path(args.candidate_output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.candidate_output).write_text(json.dumps(apply_patch(cand, patch), indent=2) + "\n", encoding="utf8")
            _print(report, args.output)
        elif args.command == "stress":
            _print(stress(load_candidate(args.candidate)), args.output)
        elif args.command == "benchmark":
            if args.random_trials < 1: raise ValueError("--random-trials must be >=1")
            records = []
            for path in sorted(Path(args.directory).glob("*.json")):
                candidate = load_candidate(path)
                records.append({"path": path.as_posix(), "author": candidate.get("author"),
                                "method": candidate.get("method"), "evaluation": evaluate(candidate),
                                "random_baseline": random_baseline(candidate, args.random_trials)})
            if not records: raise ValueError("No candidate .json files found")
            _print({"schema_version": 1, "note": "Deterministic local benchmark; not an independent blind evaluation", "records": records}, args.output)
        elif args.command == "start":
            _print(start(Path(args.session), args.task, args.mode, args.explanation, args.budget,
                         participant=args.participant, model=args.model,
                         prompt_file=Path(args.prompt_file) if args.prompt_file else None))
        elif args.command == "probe":
            _print(probe(Path(args.session), Path(args.candidate)))
        elif args.command == "finalize":
            _print(finalize(Path(args.session), Path(args.candidate)))
    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
