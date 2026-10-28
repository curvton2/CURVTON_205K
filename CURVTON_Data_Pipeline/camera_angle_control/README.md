# TO MAKE 100 CSVS INDISE EACH EASY , MEDIUM AND HARD IN THE CSV_OUTPUT DIRECTORY :

python setup_and_run.py --n 100

# Just regenerate CSVs and check them before running inference
python setup_and_run.py --csv_only

# Then verify a CSV looks right
head -3 PATH_TO_STUDIO_ROOT/csv_output/easy/easy_000.csv

# Then run inference only (CSVs already good)
python setup_and_run.py --skip_csv

# With compile (faster per-sample after ~2 min warmup)
python setup_and_run.py --compile --fa2

# One thing to be aware of — the recipe pools have ~29 (easy), ~31 (medium), ~31 (hard) unique recipes. When --n 100 exceeds the pool size, pick_n cycles and shuffles the pool to fill the remaining slots, meaning some recipes repeat across different CSVs. If you want all 100 CSVs to be truly distinct, you have two options:
# Option 1 — Accept repeats but with different seeds (current behavior). Same camera recipe on the same image with a different seed produces a slightly different output due to diffusion stochasticity. Fine for most dataset purposes.
# Option 2 — Expand the recipe pools. If you want 100 fully unique recipes per difficulty, let me know and I'll expand the prompt vocabulary — there's plenty of room (more angle increments, more combination permutations).