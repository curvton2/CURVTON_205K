
import os
import re
import shutil
import sys

# Add parent dir to path so we can import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import EDITED_DIR, INCORRECT_DIR, ULTIMATE_DIR, CORRECT_ULTIMATE_DIR, INCORRECT_ULTIMATE_DIR

def main():
    diff_levels = ["easy"]
    genders = ["male", "female"]

    # Pattern to extract human_id and original edit index
    # Example: mh_000001_e05_correct.png -> human_id: mh_000001, edit_idx: 05, status: correct
    pattern = re.compile(r"^(m|f)h_(\d+)_e(\d+)_(correct|incorrect)\.png$")

    for diff in diff_levels:
        for gender in genders:
            dir_path = os.path.join(ULTIMATE_DIR, diff, gender, "initial_person_image")
            if not os.path.exists(dir_path):
                continue

            # Group files by human_id
            human_groups = {}
            files = os.listdir(dir_path)
            
            for f in files:
                match = pattern.match(f)
                if match:
                    prefix, h_num, e_num, status = match.groups()
                    human_id = f"{prefix}h_{h_num}"
                    if human_id not in human_groups:
                        human_groups[human_id] = []
                    
                    human_groups[human_id].append({
                        "full_path": os.path.join(dir_path, f),
                        "original_name": f,
                        "e_num": int(e_num),
                        "status": status
                    })

            # Process each person's group
            for human_id, samples in human_groups.items():
                
                # 1. Handle 'incorrect' samples: Move them to dataset_ultimate_15
                for s in samples:
                    if s["status"] == "incorrect":
                        # Maintain same directory structure as dataset_ultimate_13:
                        # dataset_ultimate_15/diff/gender/initial_person_image/
                        dest_dir = os.path.join(INCORRECT_ULTIMATE_DIR, diff, gender, "initial_person_image")
                        os.makedirs(dest_dir, exist_ok=True)
                        
                        shutil.move(s["full_path"], os.path.join(dest_dir, s["original_name"]))
                        
                        # Also move the JSON prompt/feedback file if it exists
                        old_json = s["full_path"].replace(".png", ".json")
                        if os.path.exists(old_json):
                            shutil.move(old_json, os.path.join(dest_dir, s["original_name"].replace(".png", ".json")))
                        
                        print(f"Moved Incorrect: {s['original_name']} -> {dest_dir}")

                # 2. Re-filter for only the remaining 'correct' ones
                correct_samples = [s for s in samples if s["status"] == "correct" and os.path.exists(s["full_path"])]
                
                # 3. Sort them by their original eXX index to maintain generation order
                correct_samples.sort(key=lambda x: x["e_num"])

                # 4. Rename 'correct' samples consecutively (e01, e02...) and move them to dataset_ultimate_14
                # Same directory structure as dataset_ultimate_13:
                # dataset_ultimate_14/diff/gender/initial_person_image/
                for i, s in enumerate(correct_samples):
                    new_e_num = i + 1
                    new_name = f"{human_id}_e{new_e_num:02d}.png"
                    
                    dest_dir = os.path.join(CORRECT_ULTIMATE_DIR, diff, gender, "initial_person_image")
                    os.makedirs(dest_dir, exist_ok=True)
                    new_path = os.path.join(dest_dir, new_name)
                    
                    # Move companion JSON (improved prompt / eval metadata) as well
                    old_json = s["full_path"].replace(".png", ".json")
                    new_json = new_path.replace(".png", ".json")

                    try:
                        # If a file already exists with the new name (could happen if we are rerunning),
                        # usually we should skip or overwrite. We'll attempt rename.
                        if s["full_path"] != new_path:
                            shutil.move(s["full_path"], new_path)
                            if os.path.exists(old_json):
                                shutil.move(old_json, new_json)
                            print(f"Moved Correct: {s['original_name']} -> {new_path}")
                        else:
                            # Already has the correct name, just strip status
                            pass 
                    except Exception as e:
                        print(f"Error renaming {s['original_name']}: {e}")

    print("Cleanup and Renaming Complete.")

if __name__ == "__main__":
    main()
