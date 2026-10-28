# Stratified Evaluation

Try-on metrics (SSIM, PSNR, LPIPS, FID, KID) reported per stratum of the CURVTON test set. Each stratification has its own script:

| Stratification | Strata | Split built by | Evaluation script |
|---|---|---|---|
| Difficulty | `easy`, `medium`, `hard` (+ `overall`) | already in the dataset (`dataset_ultimate_test/{easy,medium,hard}/`) | `eval_difficulty.sh` |
| Body region | `upper_body`, `lower_body`, `dresses` | `stratify_by_body_region.py` | `eval_body_region.sh` |
| Traditionality | `traditional`, `non_traditional` | `stratify_by_traditionality.py` | `eval_traditionality.sh` |

All three evaluation scripts call `../zeroshot/vton_metrics/evaluate_vton.py`.

## Building the splits

```bash
python Evaluation/VTON_Evaluation/stratified/stratify_by_body_region.py \
    --curvton-root /path/to/dataset_ultimate_test \
    --output-root  /path/to/dataset_ultimate_test_body_region

python Evaluation/VTON_Evaluation/stratified/stratify_by_traditionality.py \
    --curvton-root /path/to/dataset_ultimate_test \
    --output-root  /path/to/dataset_ultimate_test_traditionality
```

For every try-on in `{easy,medium,hard}/{male,female}/tryon_image/`, the shared code in `stratify_common.py`:

1. Parses the CURVTON file name `{m|f}h_<person>_e<edit>_{m|f}c_<cloth>_<garment name>.png` and finds the matching `initial_person_image/` and `cloth_image/` files.
2. Looks the garment name up in `DRESS_INFO` (`CURVTON_assets/dictionaries/dress_info.json`, loaded through `CURVTON_Data_Pipeline/vton/tryon/core/dress_info.py`), using the same longest-suffix match as the generation pipeline.

Then:

- **Body region:** the cloth file's `_upper` / `_lower` / `_dress` suffix (set by `final_cloth_rename.py`); for cloth files that were not renamed, the garment's `dress_type` (upperbody → `upper_body`, lowerbody → `lower_body`, fullbody → `dresses`; accessories fall back to `upper_body`, as in `final_cloth_rename.py`).
- **Traditionality:** the garment's `traditional` flag in `dress_info.json`.

Each script writes:

```
<output-root>/
├── <stratum>/{male,female}/{initial_person_image,cloth_image,tryon_image}/
└── stratification_manifest.csv      # difficulty, gender, tryon_image, garment, dress_type, <stratum column>
```

Options: `--mode copy|symlink|hardlink` (default `copy`), `--prefix-difficulty` if the same file name occurs in more than one difficulty tier.

## Running the evaluations

Generated try-ons must use the same file names as the real try-ons, laid out as `<GENERATED_ROOT>/<model>/<stratum>/<male|female>/tryon_image/`.

```bash
export CURVTON_ROOT=/path/to/dataset_ultimate_test
export GENERATED_ROOT=/path/to/generated
export MODELS="catvton idmvton"

bash Evaluation/VTON_Evaluation/stratified/eval_difficulty.sh
bash Evaluation/VTON_Evaluation/stratified/eval_body_region.sh
bash Evaluation/VTON_Evaluation/stratified/eval_traditionality.sh
```

- `eval_body_region.sh` / `eval_traditionality.sh` build their split first if `BODY_REGION_ROOT` / `TRADITIONALITY_ROOT` does not exist (defaults: `${CURVTON_ROOT}_body_region`, `${CURVTON_ROOT}_traditionality`).
- Results go to `$OUTPUT_DIR/vton_metrics.{json,csv}`; defaults are `./stratified_eval_results/{difficulty,body_region,traditionality}`.
- Other settings: `SAMPLE_RATIO` (default `1.0`), `DEVICE` (default `auto`). Extra arguments go to `evaluate_vton.py`.

## Traditional vs non-traditional garments

`dress_info.json` marks garments from the regional collections (Nepal, India, Pakistan & Bangladesh, Japan, East Asia, Southeast Asia, Middle East & Central Asia, Africa, the Americas, Europe, Oceania) as `traditional: true`, and the contemporary "Modern" collection as `traditional: false`. Alias entries (alternative spellings of garment names) follow the garment they refer to.
