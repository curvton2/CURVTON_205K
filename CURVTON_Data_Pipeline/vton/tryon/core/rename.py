"""
Try-On Organize Script (Phase 3 of job_tryon_loop.sh)
=====================================================
Runs after edit_generator.py and edit_evaluator.py in each loop iteration.

Directory roles (all per {diff}/{gender}/):
    <dataset_v3>/dataset_ultimate/    working pool: pairs still waiting for an accepted try-on
        initial_person_image/  cloth_image/  tryon_image/  retry_prompt/
    <dataset_ultimate_1>/             triplets the evaluator rejected this iteration
    <dataset_end>/                    accepted final dataset (read by post_camera_edit.py
        initial_person_image/  cloth_image/  tryon_image/  eval/      and final_cloth_rename.py)
    <dataset_v3>/dataset_rejected/iter_N/   archive of every rejected try-on + its eval JSON

What this script does:
  ACCEPTED (tryon in the pool with a sidecar {name}.json written by the evaluator)
      tryon + cloth  -> moved to dataset_end
      person         -> copied to dataset_end (it may still be needed by other pairs)
      eval JSON      -> moved to dataset_end/.../eval/
  REJECTED (triplet in dataset_ultimate_1)
      tryon + eval JSON -> archived to dataset_rejected/iter_N/
      cloth + person    -> moved back into the pool
      improved_prompt   -> written to the pool's retry_prompt/{name}.json, which
                           edit_generator.py uses as the prompt in the next iteration
  Finally, pool person images whose pairs have all been accepted are removed
  (their copy already lives in dataset_end).

After the last iteration, pairs still in the pool are the ones that never passed.
"""

import argparse
import json
import os
import shutil

DIFFICULTIES = ["easy", "medium", "hard"]
GENDERS = ["male", "female"]


def person_name_for(fname: str) -> str:
    """mh_000001_e01_mc_000001_daura_suruwal.png -> mh_000001_e01.png"""
    parts = fname[:-4].split("_")
    return "_".join(parts[:3]) + ".png"


def pngs(d: str):
    return [f for f in os.listdir(d) if f.endswith(".png")] if os.path.isdir(d) else []


def move(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)


def copy_if_missing(src: str, dst: str):
    if os.path.exists(src) and not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def collect_accepted(pool: str, end: str) -> int:
    n = 0
    for fname in pngs(os.path.join(pool, "tryon_image")):
        eval_json = os.path.join(pool, "tryon_image", fname[:-4] + ".json")
        if not os.path.exists(eval_json):
            continue  # not evaluated yet
        person = person_name_for(fname)
        copy_if_missing(os.path.join(pool, "initial_person_image", person),
                        os.path.join(end, "initial_person_image", person))
        cloth_src = os.path.join(pool, "cloth_image", fname)
        if os.path.exists(cloth_src):
            move(cloth_src, os.path.join(end, "cloth_image", fname))
        move(os.path.join(pool, "tryon_image", fname), os.path.join(end, "tryon_image", fname))
        move(eval_json, os.path.join(end, "eval", fname[:-4] + ".json"))
        retry = os.path.join(pool, "retry_prompt", fname[:-4] + ".json")
        if os.path.exists(retry):
            os.remove(retry)
        n += 1
    return n


def requeue_rejected(rejected: str, pool: str, archive: str) -> int:
    n = 0
    for fname in pngs(os.path.join(rejected, "tryon_image")):
        stem = fname[:-4]
        eval_json = os.path.join(rejected, "tryon_image", stem + ".json")

        improved = ""
        if os.path.exists(eval_json):
            try:
                with open(eval_json, encoding="utf-8") as f:
                    improved = (json.load(f).get("improved_prompt") or "").strip()
            except (OSError, ValueError):
                pass
            move(eval_json, os.path.join(archive, "eval", stem + ".json"))
        move(os.path.join(rejected, "tryon_image", fname), os.path.join(archive, "tryon_image", fname))

        cloth_src = os.path.join(rejected, "cloth_image", fname)
        if os.path.exists(cloth_src):
            move(cloth_src, os.path.join(pool, "cloth_image", fname))
        person = person_name_for(fname)
        person_src = os.path.join(rejected, "initial_person_image", person)
        person_dst = os.path.join(pool, "initial_person_image", person)
        if os.path.exists(person_src):
            if os.path.exists(person_dst):
                os.remove(person_src)
            else:
                move(person_src, person_dst)

        retry = os.path.join(pool, "retry_prompt", stem + ".json")
        if improved:
            os.makedirs(os.path.dirname(retry), exist_ok=True)
            with open(retry, "w", encoding="utf-8") as f:
                json.dump({"improved_prompt": improved}, f, indent=2)
        elif os.path.exists(retry):
            os.remove(retry)  # stale prompt from an older iteration
        n += 1
    return n


def drop_orphan_persons(pool: str, end: str) -> int:
    """Remove pool persons whose pairs are all accepted (a copy is already in dataset_end)."""
    needed = {person_name_for(f) for f in pngs(os.path.join(pool, "cloth_image"))}
    n = 0
    for person in pngs(os.path.join(pool, "initial_person_image")):
        if person not in needed and os.path.exists(os.path.join(end, "initial_person_image", person)):
            os.remove(os.path.join(pool, "initial_person_image", person))
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_v3_dir", type=str, required=True,
                        help="dataset_v3 root; the working pool is <dataset_v3_dir>/dataset_ultimate")
    parser.add_argument("--dataset_ultimate_1_dir", type=str, required=True,
                        help="Where edit_evaluator.py moved rejected triplets")
    parser.add_argument("--dataset_end_dir", type=str, required=True,
                        help="Accepted final dataset")
    parser.add_argument("--iteration", type=int, default=1, help="Loop iteration (for the archive)")
    args = parser.parse_args()

    pool_root = os.path.join(args.dataset_v3_dir, "dataset_ultimate")
    archive_root = os.path.join(args.dataset_v3_dir, "dataset_rejected", f"iter_{args.iteration}")

    totals = {"accepted": 0, "requeued": 0, "orphans": 0, "pending": 0}
    for diff in DIFFICULTIES:
        for gender in GENDERS:
            pool = os.path.join(pool_root, diff, gender)
            end = os.path.join(args.dataset_end_dir, diff, gender)
            rejected = os.path.join(args.dataset_ultimate_1_dir, diff, gender)
            archive = os.path.join(archive_root, diff, gender)

            a = collect_accepted(pool, end)
            r = requeue_rejected(rejected, pool, archive)
            o = drop_orphan_persons(pool, end)
            p = len(pngs(os.path.join(pool, "cloth_image")))
            totals["accepted"] += a
            totals["requeued"] += r
            totals["orphans"] += o
            totals["pending"] += p
            print(f"[{diff}/{gender}] accepted={a} requeued={r} still_pending={p}")

    print("\n--- Summary (iteration %d) ---" % args.iteration)
    print(f"Accepted -> {args.dataset_end_dir}: {totals['accepted']}")
    print(f"Rejected -> requeued with improved prompt: {totals['requeued']}")
    print(f"Pairs still pending in the pool: {totals['pending']}")
    print(f"Unreferenced person images removed from the pool: {totals['orphans']}")


if __name__ == "__main__":
    main()
