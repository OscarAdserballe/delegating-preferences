"""S3 gate, read from a pilot's summary.json and cost projection (run analyse_results.py and estimate_cost.py first)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = (
    "accuracy.csv",
    "accuracy_tie_aware.csv",
    "regret.csv",
    "zero_regret_share.csv",
    "agreement.csv",
    "call_stats.csv",
    "results.png",
)


def main() -> int:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "pilot"
    summary = json.loads((run_dir / "summary.json").read_text())
    projection = json.loads((run_dir / "analysis" / "cost_projection.json").read_text())
    diag = summary["diagnostics"]
    unparsed = {r["arm"]: int(r["unparsed"]) for r in diag["per_arm"]}
    calls = diag["calls"]
    runs = [run_dir, *(run_dir.parent / r["run_id"] for r in summary["manifest"].get("added_runs") or [])]
    manifests = {r.name: json.loads((r / "manifest.json").read_text()) for r in runs}
    attempt = run_dir / "manifest_crusoe_attempt.json"
    if attempt.exists():
        manifests["pilot (Crusoe attempt)"] = json.loads(attempt.read_text())
    served = {name: m.get("served") for name, m in manifests.items()}
    swapped = [name for name, m in manifests.items() if "ServedModelChanged" in str(m.get("status"))]
    binds = {
        f"{c['arm']}@{c['provider']}": round(c["cap_binds_share"], 3) for c in calls if c["cap_binds_share"] is not None
    }
    length = {f"{c['arm']}@{c['provider']}": round(c["finish_length_share"], 3) for c in calls}
    cost = projection["total_usd"]
    checks = {
        f"0 unparsed (per arm: {unparsed})": sum(unparsed.values()) == 0,
        f"served provider/model stable within every run ({served})": not swapped,
        f"thinking cap binds on < 5% of calls (share with reasoning ≥ budget: {binds}; "
        f"share ending at max_tokens: {length})": all(v < 0.05 for v in binds.values()),
        f"projected main run under $20 (${cost:.2f}; ×1.25 = ${cost * 1.25:.2f})": cost * 1.25 < 20,
        "analysis produced every table and figure": all((run_dir / "analysis" / f).exists() for f in FIGURES),
    }
    for k, v in checks.items():
        print(f"  [{'ok' if v else 'FAIL'}] {k}")
    ok = all(checks.values())
    print(f"GATE S3: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
