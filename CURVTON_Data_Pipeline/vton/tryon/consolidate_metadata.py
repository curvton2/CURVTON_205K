
import os
import json
import sys

# Add parent dir to path so we can import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from config import JSONL_SRC_DIR, EDIT_JSONL_PATH
except ImportError:
    # Fallback for local testing if pathing is different
    JSONL_SRC_DIR = "vton/source_data"
    EDIT_JSONL_PATH = "vton/edit/edit.jsonl"

def consolidate_metadata():
    # Now reading from gender-specific JSONL files as requested
    src_files = [
        os.path.join(JSONL_SRC_DIR, "male.jsonl"),
        os.path.join(JSONL_SRC_DIR, "female.jsonl")
    ]
    
    out_dir = os.path.dirname(EDIT_JSONL_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    records_written = 0
    
    with open(EDIT_JSONL_PATH, 'w', encoding='utf-8') as writer:
        for src_file in src_files:
            if not os.path.exists(src_file):
                print(f"Warning: {src_file} not found. Skipping.")
                continue
            
            print(f"Reading from {src_file}...")
            with open(src_file, 'r', encoding='utf-8') as reader:
                for line in reader:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        
                        # Clean filename: "mh_000001.png" -> "mh_000001"
                        raw_fname = data.get("file_name") or data.get("filename")
                        if raw_fname:
                            clean_fname = os.path.splitext(raw_fname)[0]
                        else:
                            clean_fname = None

                        # Map to internal format for edit.jsonl
                        entry = {
                            "id": data.get("source_id"),
                            "file_name": clean_fname,
                            "gender": data.get("gender"),
                            "cloth": data.get("clothes") or data.get("cloth", []),
                            "prompt": data.get("prompt"),
                            "keywords": data.get("keywords", {})
                        }
                        
                        writer.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        records_written += 1
                    except Exception as e:
                        print(f"Skipping line in {src_file} due to error: {e}")

    print(f"Successfully consolidated {records_written} records into {EDIT_JSONL_PATH}")

if __name__ == "__main__":
    consolidate_metadata()
