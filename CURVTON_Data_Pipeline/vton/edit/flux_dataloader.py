import os
import sys
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Add parent dir to path so we can import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import HUMANS_DIR

class FluxEditDataset(Dataset):
    """
    DataLoader for Flux editing generation.
    Pairs generated .txt prompts with their source human images.
    Structure:
    - prompts: dataset_v3/edited/prompts/{difficulty}/{gender}/{human_id}_{edit_id}.txt
    - humans:  {HUMANS_DIR}/{gender}/{human_id}.png
    """
    def __init__(self, root_dir, rank=0, world_size=1):
        self.root_dir = root_dir
        self.samples = []
        
        prompts_root = os.path.join(root_dir, "edited", "prompts")
        humans_root = HUMANS_DIR
        
        if not os.path.exists(prompts_root):
            raise FileNotFoundError(f"Prompts directory not found: {prompts_root}")

        # Walk through difficulty levels (easy, medium, hard)
        for diff in ["easy", "medium", "hard"]:
            diff_path = os.path.join(prompts_root, diff)
            if not os.path.exists(diff_path): continue
            
            # Walk through genders (male, female)
            for gender in ["male", "female"]:
                gender_path = os.path.join(diff_path, gender)
                if not os.path.exists(gender_path): continue
                
                # Get all prompt files
                files = sorted([f for f in os.listdir(gender_path) if f.endswith(".txt")])
                
                for f in files:
                    # mh_000001_e01.txt -> human_id is mh_000001, edit_id is e01
                    prompt_path = os.path.join(gender_path, f)
                    base_name = f.replace(".txt", "")
                    parts = base_name.split("_")
                    
                    if len(parts) < 3: continue # Expecting prefix_number_editID
                    
                    human_id = f"{parts[0]}_{parts[1]}" # e.g. mh_000001
                    source_img_path = os.path.join(humans_root, gender, f"{human_id}.png")
                    
                    if os.path.exists(source_img_path):
                        # Construct the target output path for the generated image
                        output_img_path = os.path.join(root_dir, "edited", diff, gender, f"{base_name}.png")
                        
                        self.samples.append({
                            "prompt_path": prompt_path,
                            "source_img_path": source_img_path,
                            "output_img_path": output_img_path,
                            "base_name": base_name,
                            "difficulty": diff,
                            "gender": gender
                        })

        # Sharding for multi-GPU Flux inference
        self.my_samples = [s for i, s in enumerate(self.samples) if i % world_size == rank]
        print(f"Total pairs found: {len(self.samples)} | Shard size (Rank {rank}): {len(self.my_samples)}")

    def __len__(self):
        return len(self.my_samples)

    def __getitem__(self, idx):
        s = self.my_samples[idx]
        
        # Load and resize the source human image to 1024x1024 for Flux 2 Klein
        source_image = Image.open(s["source_img_path"]).convert("RGB").resize((1024, 1024))
        
        # Load the edit prompt
        with open(s["prompt_path"], 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
            
        return {
            "image": source_image, # reference image for identity/body
            "prompt": prompt,
            "output_path": s["output_img_path"],
            "base_name": s["base_name"]
        }

def custom_collate(batch):
    return {
        "image": [item["image"] for item in batch],
        "prompt": [item["prompt"] for item in batch],
        "output_path": [item["output_path"] for item in batch],
        "base_name": [item["base_name"] for item in batch]
    }

def get_edit_dataloader(root_dir, batch_size=1, num_workers=4, rank=0, world_size=1):
    dataset = FluxEditDataset(root_dir, rank=rank, world_size=world_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=custom_collate
    )

if __name__ == "__main__":
    # Example validation
    import sys
    ROOT = "dataset_v3"
    try:
        loader = get_edit_dataloader(ROOT, batch_size=4)
        for i, batch in enumerate(loader):
            print(f"Batch {i}: {len(batch['prompt'])} prompts ready for Flux.")
            if i >= 0: break
    except Exception as e:
        print(f"Error during dataloader test: {e}")
