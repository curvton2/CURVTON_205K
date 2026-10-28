
import os
import json
import torch
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import FluxPipeline # Or any other VTON model specified
from matching_rules import is_compatible
from config import IMPROVED_PROMPTS_DIR

# Multi-GPU Environment
RANK = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))

class VTONGenerationDataset(Dataset):
    """
    Dataloader that pairs Edited Human Images with Clothing Images.
    Uses edit.jsonl to ensure compatible pairings.
    """
    def __init__(self, root_dir, jsonl_path, rank=0, world_size=1):
        self.root_dir = root_dir
        self.samples = []
        
        # 1. Load Attribute Metadata
        # Mapping: human_base_id (mh_000001) -> [list of clothes]
        self.attribute_map = {}
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        fid = data.get("file_name") # e.g. mh_000001
                        if fid:
                            self.attribute_map[fid] = data.get("cloth", [])
                    except: continue
        else:
            print(f"Warning: Metadata {jsonl_path} not found.")

        # 2. Discover Clothes
        clothes = {"male": [], "female": []}
        for g in ["male", "female"]:
            p = os.path.join(root_dir, "clothes", g)
            if os.path.exists(p):
                # mc_000101.png
                clothes[g] = sorted([f for f in os.listdir(p) if f.endswith(".png")])

        # 3. Discover Edited Person Images
        # Structure: edited/{diff}/{gender}/{mh_000001_e01.png}
        edited_root = os.path.join(root_dir, "edited")
        all_possible_pairs = []
        
        for diff in ["easy", "medium", "hard"]:
            for gender in ["male", "female"]:
                dir_path = os.path.join(edited_root, diff, gender)
                if not os.path.exists(dir_path): continue
                
                person_files = sorted([f for f in os.listdir(dir_path) if f.endswith(".png") and not ("_correct" in f or "_incorrect" in f)])
                
                for pf in person_files:
                    # Parse filename: mh_000001_e01.png
                    base_parts = pf.replace(".png", "").split("_")
                    if len(base_parts) < 3: continue
                    
                    human_base_id = f"{base_parts[0]}_{base_parts[1]}" # mh_000001
                    edit_id = base_parts[2] # e01
                    
                    currently_worn = self.attribute_map.get(human_base_id, [])
                    
                    # Pair with every cloth of same gender
                    for cf in clothes[gender]:
                        # Naming: mc_000101.png -> 000101
                        cloth_id_short = cf.replace(".png", "").split("_")[-1]
                        
                        # In a real scenario, we'd look up the cloth's category (e.g., T-shirt)
                        # For now, we'll simulate this by map or keyword search in filename
                        # This is where 'matching_rules' prevents 'unusual pairings'
                        target_category = "T-shirt" # Placeholder: in production, load from clothes metadata
                        
                        if is_compatible(currently_worn, target_category):
                            all_possible_pairs.append({
                                "person_path": os.path.join(dir_path, pf),
                                "cloth_path": os.path.join(root_dir, "clothes", gender, cf),
                                "out_dir": os.path.join(root_dir, "tryon", diff, gender),
                                "human_id": base_parts[1],
                                "cloth_id": cloth_id_short,
                                "edit_id": edit_id,
                                "gender_pref": "mt" if gender == "male" else "ft"
                            })

        # 4. Shard
        self.my_samples = [s for i, s in enumerate(all_possible_pairs) if i % world_size == rank]
        print(f"Rank {rank}: Total compatible pairs: {len(all_possible_pairs)} | My Shard: {len(self.my_samples)}")

    def __len__(self):
        return len(self.my_samples)

    def __getitem__(self, idx):
        s = self.my_samples[idx]
        
        # Construct filename: mt_000001_c000101_e01.png
        out_name = f"{s['gender_pref']}_{s['human_id']}_c{s['cloth_id']}_{s['edit_id']}.png"
        output_path = os.path.join(s["out_dir"], out_name)

        # 1. Skip if already in ultimate
        # Construct the ultimate path to check
        ultimate_path = output_path.replace("dataset_v3" + os.sep + "tryon", "dataset_ultimate")
        if os.path.exists(ultimate_path):
            return {"skip": True, "path": output_path}

        p_img = Image.open(s["person_path"]).convert("RGB")
        c_img = Image.open(s["cloth_path"]).convert("RGB")
        
        # 2. Check for Improved Prompt from Scorer (Previous Iteration)
        prompt_id = out_name.replace(".png", "")
        improved_prompt_path = os.path.join(IMPROVED_PROMPTS_DIR, f"{prompt_id}.txt")
        
        if os.path.exists(improved_prompt_path):
            with open(improved_prompt_path, 'r', encoding='utf-8') as f:
                generation_prompt = f.read().strip()
        else:
            generation_prompt = "Make this person tryon this cloth"
        
        return {
            "skip": False,
            "person_image": p_img,
            "cloth_image": c_img,
            "prompt": generation_prompt,
            "output_path": output_path
        }

import concurrent.futures

class AsyncImageWriter:
    def __init__(self, max_workers=4):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []

    def save(self, img, path):
        def _save():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            img.save(path)
        self.futures.append(self.executor.submit(_save))

    def wait(self):
        concurrent.futures.wait(self.futures)
        self.futures = []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="dataset_v3")
    parser.add_argument("--jsonl", type=str, default="vton/edit/edit.jsonl")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_limit", type=int, default=-1, help="Total samples to process for testing")
    args = parser.parse_args()

    # Calculate local limit for testing
    local_limit = -1
    if args.test_limit > 0:
        local_limit = args.test_limit // WORLD_SIZE
        if RANK < (args.test_limit % WORLD_SIZE):
            local_limit += 1

    # 1. Device
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device(f"cuda:{LOCAL_RANK}")

    # 2. Pipeline (Placeholder)
    print(f"Loading VTON Model from {args.model_path}...")
    # pipe = VTONPipeline.from_pretrained(args.model_path).to(device)

    # 3. Dataloader
    dataset = VTONGenerationDataset(args.root_dir, args.jsonl, rank=RANK, world_size=WORLD_SIZE)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    writer = AsyncImageWriter()

    processed_count = 0
    # 4. Loop
    for i, item in enumerate(loader):
        if item.get("skip", [False])[0]:
            continue
            
        if local_limit > 0 and processed_count >= local_limit:
            print(f"[Rank {RANK}] Reached local test limit of {local_limit}. Terminating.")
            break
            
        # Generation Logic
        # result = pipe(prompt=item['prompt'], image=item['person_image'], garment=item['cloth_image'])
        # result_img = result.images[0]
        
        # ASYNC SAVE
        # writer.save(result_img, item['output_path'][0])
        
        processed_count += 1
        if i % 10 == 0:
            print(f"[Rank {RANK}] Generated {item['output_path'][0]} (Item {i})")

    print(f"[Rank {RANK}] Waiting for final writes...")
    writer.wait()


if __name__ == "__main__":
    main()

