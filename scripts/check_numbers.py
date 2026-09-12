"""Fail if a headline number quoted in the documents disagrees with docs/img/numbers.json.

    uv run python scripts/check_numbers.py

The figure script is the only producer of numbers; this is the check that the prose kept up.
Each entry names a value and the text that must appear verbatim in the write-up (or README).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NUMBERS = Path("docs/img/numbers.json")
DOCS = {"writeup": Path("docs/writeup.md"), "readme": Path("README.md")}


def expectations(n: dict) -> list[tuple[str, str, str]]:
    """(label, document, text that must appear)."""
    e19 = n["eval_2019"]["seeds_5_9"]
    det, base = e19["detector"], e19["baseline"]
    fa = det["clean"]["fa_per_week"]
    exp: list[tuple[str, str, str]] = [
        ("calibration factor", "writeup", f"{n['calibration_factor']:.2f}"),
        ("clean FA/week 2019", "writeup", f"{fa:.2f} false alarms"),
        ("coverage 2019", "writeup", f"{det['clean']['coverage']:.2f}"),
        ("baseline FA/week", "writeup", f"{base['clean']['fa_per_week']:.2f}"),
        ("trough hour", "writeup", f"{n['hour_min']:02d}:00"),
        ("peak hour", "writeup", f"{n['hour_max']:02d}:00"),
        ("weekday mean", "writeup", f"{n['weekday_mean_kw']:.1f} kW"),
        ("temperature r", "writeup", f"r = {n['temp_r']:.2f}"),
        ("replay alert total", "writeup", f"{n['replay_alerts_total']}"),
        ("baseline matched k", "writeup", f"k = {n['baseline_matched_k']}"),
        ("chance recall", "writeup", f"{n['offset_sweep_6h'][0]['recall']:.2f}"),
        ("cal MAE", "writeup", f"{n['cal_mae']:.1f} kW"),
        ("r kw cdd", "writeup", f"{n['r_kw_cdd_train']:.3f}"),
    ]
    for yr, cov in n["coverage_by_year"].items():
        exp.append((f"coverage {yr} (frozen model)", "writeup", f"{float(cov):.2f}"))
    for kind, row in det.items():
        if kind != "clean":
            exp.append((f"recall {kind}", "writeup", f"{row['recall']:.2f}"))
    return exp


def main() -> int:
    n = json.loads(NUMBERS.read_text())
    text = {k: p.read_text() for k, p in DOCS.items()}
    missing = [(lab, doc, s) for lab, doc, s in expectations(n) if s not in text[doc]]
    for lab, doc, s in missing:
        print(f"MISSING in {doc}: {lab} = {s!r}")
    print(f"{len(expectations(n)) - len(missing)} of {len(expectations(n))} numbers found")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
