
import os
import json
import torch
import shutil
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from config import SCORER_SYSTEM_PROMPT, IMPROVED_PROMPTS_DIR

class VTONDataset(Dataset):
    def __init__(self, root_dir, jsonl_dir, rank=0, world_size=1):
        self.root_dir = root_dir
        all_samples = []
        self.attribute_map = self._load_attribute_map(jsonl_dir)
        
        # Discover all tryon images
        tryon_root = os.path.join(root_dir, "tryon")
        for diff in ["easy", "medium", "hard"]:
            for gender in ["male", "female"]:
                dir_path = os.path.join(tryon_root, diff, gender)
                if not os.path.exists(dir_path): continue
                
                # Sorted to ensure sequential consistency across all ranks
                files = sorted(os.listdir(dir_path))
                for fname in files:
                    if fname.endswith(".png") and not ("_correct" in fname or "_incorrect" in fname):
                        all_samples.append({
                            "fname": fname,
                            "path": os.path.join(dir_path, fname),
                            "difficulty": diff,
                            "gender": gender
                        })
        
        # Shard the discovered samples sequentially by rank
        self.samples = [s for i, s in enumerate(all_samples) if i % world_size == rank]


    def _load_attribute_map(self, jsonl_dir):
        """Maps human_id (e.g. mh_00001) to its cloth keywords."""
        attr_map = {}
        if not os.path.exists(jsonl_dir):
            print(f"Warning: JSONL directory {jsonl_dir} not found.")
            return attr_map
            
        for f in os.listdir(jsonl_dir):
            if f.endswith(".jsonl"):
                with open(os.path.join(jsonl_dir, f), 'r', encoding='utf-8') as jf:
                    for line in jf:
                        data = json.loads(line)
                        # Assuming 'file_name' in JSONL corresponds to 'mh_00001' part
                        fid = data.get("file_name", "")
                        if fid:
                            attr_map[fid] = data.get("cloth", [])
        return attr_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        fname = s["fname"]
        # mt_000001_c00101_e01.png
        parts = fname.replace(".png", "").split("_")
        
        human_id_num = parts[1]
        cloth_id_num = parts[2].replace("c", "")
        edit_id = parts[3]
        
        gender_prefix = "mh" if s["gender"] == "male" else "fh"
        cloth_prefix = "mc" if s["gender"] == "male" else "fc"
        
        human_id = f"{gender_prefix}_{human_id_num}"
        cloth_id = f"{cloth_prefix}_{cloth_id_num}"
        
        # Paths to referenced images
        edited_path = os.path.join(self.root_dir, "edited", s["difficulty"], s["gender"], f"{human_id}_{edit_id}.png")
        cloth_path = os.path.join(self.root_dir, "clothes", s["gender"], f"{cloth_id}.png")
        
        # Get attributes of person from map
        original_clothes = self.attribute_map.get(human_id, [])
        
        # Load tryon image
        image = Image.open(s["path"]).convert("RGB")
        
        user_msg = f"Evaluate this Virtual Try-On image.\n"
        user_msg += f"Person was wearing: {', '.join(original_clothes)}\n"
        user_msg += f"Target garment to try on: {cloth_id}\n"
        user_msg += f"The generation prompt used was: 'Make this person tryon this cloth'\n"
        
        return {
            "image": image,
            "user_prompt": user_msg,
            "sample": s,
            "human_id": human_id,
            "cloth_id": cloth_id
        }

import concurrent.futures

class AsyncScorerWriter:
    def __init__(self, max_workers=8):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []

    def handle_correct(self, src, dst):
        def _task():
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
        self.futures.append(self.executor.submit(_task))

    def handle_incorrect(self, src, dst, prompt_path, prompt):
        def _task():
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt)
        self.futures.append(self.executor.submit(_task))

    def wait(self):
        concurrent.futures.wait(self.futures)
        self.futures = []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Root folder structure path")
    parser.add_argument("--jsonl_dir", type=str, required=True, help="Path to JSONL attribute files")
    parser.add_argument("--ultimate_dir", type=str, default="dataset_ultimate", help="Directory for correct try-ons")
    parser.add_argument("--incorrect_dir", type=str, default="try_on_incorrect_directory", help="Directory for failed try-ons")
    parser.add_argument("--iteration", type=int, default=1, help="Current iteration number")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--test_limit", type=int, default=-1, help="Total samples to process for testing")
    args = parser.parse_args()

    # Calculate local limit for testing
    local_limit = -1
    RANK = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    
    if args.test_limit > 0:
        local_limit = args.test_limit // WORLD_SIZE
        if RANK < (args.test_limit % WORLD_SIZE):
            local_limit += 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model {args.model_name}...")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()

    dataset = VTONDataset(args.root_dir, args.jsonl_dir, rank=RANK, world_size=WORLD_SIZE)
    print(f"[Rank {RANK}] Found {len(dataset)} try-on samples to score.")

    os.makedirs(IMPROVED_PROMPTS_DIR, exist_ok=True)
    os.makedirs(args.ultimate_dir, exist_ok=True)
    os.makedirs(args.incorrect_dir, exist_ok=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    writer = AsyncScorerWriter()

    processed_count = 0
    for i, item in enumerate(loader):
        if local_limit > 0 and processed_count >= local_limit:
            print(f"[Rank {RANK}] Reached local test limit of {local_limit}. Terminating.")
            break
        
        # s = item["sample"] # Note: DataLoader colates this, so we access via index 0 for batch_size=1
        s = {k: v[0] if isinstance(v, list) else v.item() if torch.is_tensor(v) else v for k,v in item['sample'].items()}

        
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": SCORER_SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": item["image"]},
                {"type": "text", "text": item["user_prompt"]}
            ]}
        ]
        
        inputs = processor.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=512)
        
        response = processor.decode(output[0], skip_special_tokens=True)
        
        try:
            if "```json" in response:
                json_str = response.split("```json")[-1].split("```")[0].strip()
            else:
                json_str = response.strip()
            
            res_json = json.loads(json_str)
            label = res_json.get("label", "INCORRECT").upper()
            
            if label == "CORRECT":
                out_path = os.path.join(args.ultimate_dir, s["difficulty"], s["gender"], s["fname"])
                writer.handle_correct(s["path"], out_path)
            else:
                name_base, ext = os.path.splitext(s["fname"])
                out_name = f"{name_base}_iter{args.iteration}{ext}"
                out_path = os.path.join(args.incorrect_dir, s["difficulty"], s["gender"], out_name)
                prompt_path = os.path.join(IMPROVED_PROMPTS_DIR, f"{name_base}.txt")
                writer.handle_incorrect(s["path"], out_path, prompt_path, res_json.get("corrected_prompt", ""))
            
            processed_count += 1
            if i % 10 == 0:
                print(f"[Rank {RANK}] Scored {s['fname']} -> {label}")
            
        except Exception as e:
            print(f"Error processing {s['fname']}: {e}")

    writer.wait()

if __name__ == "__main__":
    main()
