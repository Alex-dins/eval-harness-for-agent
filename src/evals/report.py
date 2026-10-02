"""Reporting: aggregate CaseResults into a summary, a console table, and JSON.

Kept separate from the runner so you can re-render or re-aggregate stored results
without re-running the (paid) agent.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .schema import CaseResult


def summarize(results: list[CaseResult]) -> dict:
    """Pass counts per category plus an overall total."""
    summary: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = summary.setdefault(r.category, {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(r.passed)

    total = sum(b["total"] for b in summary.values())
    passed = sum(b["passed"] for b in summary.values())
    summary["overall"] = {"passed": passed, "total": total}
    return summary


def _failed_check_names(result: CaseResult) -> str:
    failed = [c.name for c in result.checks if not c.passed]
    return ",".join(failed)


def print_table(results: list[CaseResult]) -> None:
    """Print a human-readable table of per-case outcomes, then the summary."""
    id_w = max((len(r.case_id) for r in results), default=4)
    id_w = max(id_w, len("case"))
    cat_w = max((len(r.category) for r in results), default=8)

    header = f"{'case':<{id_w}}  {'category':<{cat_w}}  result  failed_checks"
    print(header)
    print("-" * len(header))
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(
            f"{r.case_id:<{id_w}}  {r.category:<{cat_w}}  {mark:<6}  "
            f"{_failed_check_names(r)}"
        )

    print()
    summary = summarize(results)
    overall = summary.pop("overall")
    for category, b in summary.items():
        print(f"{category:<14} {b['passed']}/{b['total']}")
    print("-" * 22)
    print(f"{'overall':<14} {overall['passed']}/{overall['total']}")


def write_json(results: list[CaseResult], out_dir: Path | str = "results") -> Path:
    """Write full results (including trajectories) to results/<timestamp>.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{stamp}.json"

    payload = {
        "summary": summarize(results),
        "results": [r.model_dump() for r in results],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def all_passed(results: list[CaseResult]) -> bool:
    return all(r.passed for r in results)
