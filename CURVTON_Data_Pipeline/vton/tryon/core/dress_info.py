# DRESS_INFO — Garment metadata for virtual try-on (VTON) pipeline
#
# dress_type categories:
#   "fullbody dress"        — robes, sarees, kimonos, gowns (cover torso + legs as one)
#   "lowerbody dress"       — skirts, trousers, wraps secured at waist
#   "upperbody inner dress" — standalone tops, shirts, blouses, tunics up to knee length
#   "upperbody flex dress"  — outer layers worn OVER inner: jackets, coats, sweaters
#   "fullbody flex dress"   — full-length outer layers worn OVER a complete outfit
#                             (cloaks, ceremonial over-robes, draped full-body wraps)
#   "shawl"                 — rectangular/square cloth draped over shoulders or full body
#   "scarf"                 — cloth wrapped around the head or neck
#   "cap"                   — structured hats and headwear
#   "head wrap"             — fabric wrapped around head in a decorative/formal style
#   "head and face wrap"    — fabric covering both head and face
#
# description rules:
#   - Structural features ONLY: silhouette, length, sleeve type, collar, closures
#   - NO color, NO pattern, NO embroidery or surface decoration
#   - Max 30 words
#
# prompt template:
#   "Make this person wear this dress, which is a {name},
#    a {dress_type} characterized by its {description}."

# The garment list itself lives in CURVTON_assets/dictionaries/dress_info.json.

import json
from pathlib import Path

_DRESS_INFO_JSON = Path(__file__).resolve().parents[4] / "CURVTON_assets" / "dictionaries" / "dress_info.json"
with open(_DRESS_INFO_JSON, encoding="utf-8") as _f:
    DRESS_INFO = json.load(_f)
