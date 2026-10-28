
import os

# ==============================================================================
# 1. DIRECTORY STRUCTURE & PATHS
# ==============================================================================
VTON_PARENT_DIR = "PATH_TO_VTON_PARENT_DIR"

ROOT_DIR = "PATH_TO_DATASET_ROOT"
# Source humans are located in the human_gen project folder
HUMANS_DIR = os.path.join(ROOT_DIR, "humans")
CLOTHES_DIR = os.path.join(ROOT_DIR, "clothes")
EDITED_DIR = os.path.join(ROOT_DIR, "edited")
PROMPTS_DIR = os.path.join(EDITED_DIR, "prompts")
TRYON_DIR = os.path.join(ROOT_DIR, "tryon")

# Source dataset for evaluation (generations to be evaluated)
ULTIMATE_DIR = os.path.join(ROOT_DIR, "dataset_ultimate_13")

# Output for high-quality accepted (correct) results — same structure as dataset_ultimate_13
CORRECT_ULTIMATE_DIR = os.path.join(ROOT_DIR, "dataset_ultimate_14")

# Output for incorrect/rejected results — same structure as dataset_ultimate_13
INCORRECT_ULTIMATE_DIR = os.path.join(ROOT_DIR, "dataset_ultimate_15")

# Storage for failed results and improved prompts (legacy)
INCORRECT_DIR = os.path.join(ROOT_DIR, "try_on_incorrect_directory")
IMPROVED_PROMPTS_DIR = "vton/improved_prompts"

# Metadata
EDIT_JSONL_PATH = os.path.join(VTON_PARENT_DIR, "edit.jsonl")
VTON_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSONL_SRC_DIR = os.path.join(VTON_BASE_DIR, "source_data")
ASSETS_DIR = os.path.abspath(os.path.join(VTON_BASE_DIR, "..", "..", "CURVTON_assets"))
SYSTEM_PROMPTS_DIR = os.path.join(ASSETS_DIR, "system_prompts")
CAMERA_CONTROL_JSON = os.path.join(ASSETS_DIR, "dictionaries", "camera_control.json")

# ==============================================================================
# 2. MODEL PATHS
# ==============================================================================
MODEL_PATH_QWEN = "Qwen/Qwen3-VL-32B-Instruct"
MODEL_PATH_FLUX = "black-forest-labs/FLUX.2-klein-9B"
MODEL_PATH_VTON = "PATH_TO_VTON_MODEL" # Update this to your local VTON model path

# ==============================================================================
# 3. SYSTEM PROMPTS
# ==============================================================================

# Prompt for Phase 1.1: Generating Flux instructions from keywords
EDIT_SYSTEM_PROMPT_FILE = os.path.join(SYSTEM_PROMPTS_DIR, "easy_edit_system_prompt.txt")
try:
    with open(EDIT_SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as _f:
        EDIT_SYSTEM_PROMPT = _f.read().strip()
except Exception:
    EDIT_SYSTEM_PROMPT = ""

# Prompt for Phase 1.4: Evaluating Flux edits (QC) — accepts at score >= 0.8
EDIT_EVAL_PROMPT_FILE = os.path.join(SYSTEM_PROMPTS_DIR, "edited_person_qc_prompt.txt")

TRYON_EVAL_PROMPT_FILE = os.path.join(SYSTEM_PROMPTS_DIR, "tryon_feedback_prompt.txt")

# Prompt for Phase 2.2: Scoring Try-On results and generating improved feedback
SCORER_SYSTEM_PROMPT = """You are a Virtual Try-On (VTON) quality evaluator. 
Given a try-on image, assess if it looks realistic and accurately reflects the target garment being worn by the person.
Rate the image as either "CORRECT" (looks natural, high quality, correct garment) or "INCORRECT" (artifacts, unnatural fit, wrong garment).
If "INCORRECT", you must provide a "corrected_prompt" that describes the scene and fit better to guide a diffusion model.

Output your evaluation ONLY in the following JSON format:
{
  "label": "CORRECT" or "INCORRECT",
  "reason": "Brief explanation",
  "corrected_prompt": "Refined prompt for better VTON generation"
}
"""
