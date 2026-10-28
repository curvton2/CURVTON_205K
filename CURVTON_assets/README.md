# CURVTON Assets

System prompts and dictionaries used by the CURVTON data-generation pipeline. Code loads these files directly (paths are resolved relative to each script), so edit them here.

## `docs/images/`

Figures from the WACV paper used in the main README: `teaser.jpg`, `pipeline.png` (data-generation pipeline), `tier_easy.jpg`, `tier_medium.jpg`, `tier_hard.jpg`.

## `human_eval_rubrics.json`

The 10-question rubric (Q1–Q10) annotators use in the human evaluation of try-on results: object placement, background consistency, structural details, text/logo fidelity, artifacts and blending, unwanted object introduction, anatomical correctness, garment fit and drape realism, overall try-on plausibility, and an open-ended question. Each entry has `id`, `criterion` and `check` (what the annotator checks).

## `dictionaries/`

| File | Contents | Used by |
|---|---|---|
| `cloth_dictionary.json` | 100 garment entries sampled for cloth generation (Cloth Stream) | `CURVTON_Data_Pipeline/cloth_stream/generate_cloth.py` |
| `person_dictionary.json` | Body types (9), poses, country/ethnicity groups (14), gender — with sampling probabilities (Person Stream) | reference for person prompt generation |
| `dress_info.json` | 385 garments with `name`, `dress_type`, structural `description` and a `traditional` flag (237 traditional, 148 non-traditional); builds the try-on prompt and the traditionality split | `CURVTON_Data_Pipeline/vton/tryon/core/*` (via `dress_info.py`), `Evaluation/VTON_Evaluation/stratified/stratify_by_traditionality.py`, `stratify_by_body_region.py` |
| `camera_control.json` | Camera-angle edit prompts, 100 per difficulty (easy / medium / hard) | `camera_angle_control/`, `vton/sample_prompts.py`, `vton/tryon/core/post_camera_edit.py` |

## `system_prompts/`

The first three prompts are reproduced exactly as in the supplementary material (Appendix "System Prompts Used in the Generation and Evaluation Pipeline") and are named after their listings there.

| File | Paper listing | Step | Used by |
|---|---|---|---|
| `medium_hard_person_edit_prompt.txt` | Difficulty-Aware Person Editing Prompt (`lst:person_edit_prompt`) | Qwen3-VL: keywords + person image → FLUX.2 Klein 9B edit instruction (medium / hard) | `CURVTON_Data_Pipeline/shared/qwen_edit_prompts_core.py` |
| `edited_person_qc_prompt.txt` | Edited-Person Quality-Control Prompt (`lst:edited_person_qc_prompt`) | Qwen3-VL: accept / reject an edited person image (ACCEPT at score ≥ 0.8) | `vton/config.py` → `vton/edit/edit_evaluator.py` |
| `tryon_feedback_prompt.txt` | VLM-Based Try-On Evaluation and Feedback Prompt (`lst:tryon_feedback_prompt`) | Qwen3-VL: score a try-on triplet (person, garment, result) and write an improved prompt (SUCCESS at score ≥ 0.6) | `vton/config.py` → `vton/tryon/core/edit_evaluator.py` |
| `easy_edit_system_prompt.txt` | — | Qwen3-VL: easy-tier edit instruction | `vton/config.py` → `vton/edit/generate_edit_prompts.py` |
| `person_prompt_to_json_system_prompt.txt` | — | Qwen3-VL: person description → structured JSON | `CURVTON_Data_Pipeline/person_stream/person_prompts_to_json.py` |
