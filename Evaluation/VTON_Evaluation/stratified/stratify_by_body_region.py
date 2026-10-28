#!/usr/bin/env python3
"""Split a CURVTON test set by the body region the garment covers.

    upper_body | lower_body | dresses

The cloth file's _upper / _lower / _dress suffix (added by final_cloth_rename.py)
decides the region. If a cloth file was not renamed, the garment's dress_type in
DRESS_INFO is used instead: upperbody -> upper_body, lowerbody -> lower_body,
fullbody -> dresses; other types (shawl, scarf, cap, head wrap) fall back to
upper_body, as in final_cloth_rename.py.

Usage:
    python Evaluation/VTON_Evaluation/stratified/stratify_by_body_region.py \\
        --curvton-root /path/to/dataset_ultimate_test \\
        --output-root  /path/to/dataset_ultimate_test_body_region
"""

from stratify_common import run

REGION_BY_SUFFIX = {"upper": "upper_body", "lower": "lower_body", "dress": "dresses"}


def body_region(row: dict) -> str:
    suffix = row["cloth_image"].stem.rsplit("_", 1)[-1].lower()
    if suffix in REGION_BY_SUFFIX:
        return REGION_BY_SUFFIX[suffix]
    dress_type = row["garment"]["dress_type"].lower()
    if "upperbody" in dress_type:
        return "upper_body"
    if "lowerbody" in dress_type:
        return "lower_body"
    if "fullbody" in dress_type:
        return "dresses"
    return "upper_body"


if __name__ == "__main__":
    raise SystemExit(run(__doc__.split("\n\n")[0], body_region, "body_region"))
