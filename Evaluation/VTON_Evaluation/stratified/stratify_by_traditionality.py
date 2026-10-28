#!/usr/bin/env python3
"""Split a CURVTON test set into traditional and non-traditional garments.

    traditional | non_traditional

Uses the garment's "traditional" flag in DRESS_INFO
(CURVTON_assets/dictionaries/dress_info.json): regional collections are
traditional, the contemporary "Modern" collection is non-traditional.

Usage:
    python Evaluation/VTON_Evaluation/stratified/stratify_by_traditionality.py \\
        --curvton-root /path/to/dataset_ultimate_test \\
        --output-root  /path/to/dataset_ultimate_test_traditionality
"""

from stratify_common import run


def traditionality(row: dict) -> str:
    return "traditional" if row["garment"].get("traditional") else "non_traditional"


if __name__ == "__main__":
    raise SystemExit(run(__doc__.split("\n\n")[0], traditionality, "traditionality"))
