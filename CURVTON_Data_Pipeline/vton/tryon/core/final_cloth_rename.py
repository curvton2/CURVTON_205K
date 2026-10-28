"""
Final Cloth Rename
==================
Runs at the very end of the try-on pipeline.
Renames cloth files in dataset_ultimate from their full garment category 
names (e.g. mh_000001_e01_mc00101_v_neck_t_shirt.png) cleanly to:
- _upper.png
- _lower.png
- _dress.png
"""

import os
import sys

# Load from dress_info.py in the tryon directory
VTON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(VTON_ROOT, 'tryon'))

try:
    from dress_info import DRESS_INFO
except ImportError:
    print("CRITICAL: dress_info.py not found.")
    sys.exit(1)

def get_base_type(dress_type: str) -> str:
    dt = dress_type.lower()
    if 'upperbody' in dt:
        return 'upper'
    elif 'lowerbody' in dt:
        return 'lower'
    elif 'fullbody' in dt:
        return 'dress'
    else:
        return 'upper' # Default fallback if missing

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Root dataset directory")
    parser.add_argument("--dataset_end_dir", type=str, required=True, help="Path to ultimate dataset")
    args = parser.parse_args()

    ultimate_dir = args.dataset_end_dir
    
    if not os.path.exists(ultimate_dir):
        print(f"Directory {ultimate_dir} not found.")
        return

    print("======================================================")
    print("PHASE 5: FINAL CLOTH RENAMING")
    print("======================================================")

    total_renamed = 0
    total_skipped = 0

    for diff in ["easy", "medium", "hard"]:
        for gender in ["male", "female"]:
            cloth_dir = os.path.join(ultimate_dir, diff, gender, "cloth_image")
            if not os.path.exists(cloth_dir):
                continue
                
            for file_name in os.listdir(cloth_dir):
                if not file_name.endswith(".png"):
                    continue
                
                # We need to find the longest matching dress name from the end of the filename
                name_no_ext = file_name.replace(".png", "").lower()
                
                best_match = None
                best_name_len = -1
                found_name_underscored = ""
                
                for d in DRESS_INFO:
                    d_name_underscored = d["name"].replace(" ", "_").lower()
                    
                    if name_no_ext.endswith(d_name_underscored):
                        if len(d_name_underscored) > best_name_len:
                            best_match = d
                            best_name_len = len(d_name_underscored)
                            found_name_underscored = d_name_underscored
                            
                if best_match:
                    new_type = get_base_type(best_match["dress_type"])
                    
                    # Cut off the "_{cloth_name}" from the end of the filename
                    orig_no_ext = file_name.replace(".png", "")
                    cut_idx = name_no_ext.rfind(f"_{found_name_underscored}")
                    if cut_idx != -1:
                        base_name = orig_no_ext[:cut_idx]
                    else:
                        base_name = orig_no_ext
                        
                    new_filename = f"{base_name}_{new_type}.png"
                    
                    old_path = os.path.join(cloth_dir, file_name)
                    new_path = os.path.join(cloth_dir, new_filename)
                    
                    if old_path != new_path:
                        os.rename(old_path, new_path)
                        total_renamed += 1
                    else:
                        total_skipped += 1 # Already correctly named
                else:
                    # Could not parse dress type, skip
                    total_skipped += 1

    print(f"\nFinal Cleanup complete.")
    print(f"Successfully renamed : {total_renamed} files.")
    print(f"Skipped / No Match   : {total_skipped} files.")

if __name__ == "__main__":
    main()
