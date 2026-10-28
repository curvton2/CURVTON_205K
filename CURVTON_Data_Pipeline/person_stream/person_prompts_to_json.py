"""
Qwen3-VL Humans Dataset Processor (TEXT-ONLY INFERENCE)
======================================================
Processes human prompts (no images to VLM), extracting structured JSON.
Handles re-naming (hm_XXXXX, hf_XXXXX) and distributed processing.
Sequential sharding and asynchronous disk I/O for max performance.
"""

import argparse
import json
import os
import queue
import sys
import threading
import time
import torch
import shutil
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

# Optimized for performance
sys.stdout.reconfigure(line_buffering=True)

RANK       = int(os.environ.get("SLURM_PROCID", "0"))
LOCAL_RANK = int(os.environ.get("SLURM_LOCALID", "0"))
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", "1"))

def log(msg: str):
    print(f"[GPU {LOCAL_RANK} | RANK {RANK}] {msg}", flush=True)

SYSTEM_PROMPT = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                          "CURVTON_assets", "system_prompts", "person_prompt_to_json_system_prompt.txt"), encoding="utf-8").read()

class HumansDataset(Dataset):
    def __init__(self, root_dir, rank, world_size):
        self.root_dir = root_dir
        self.all_samples = []
        for gender in ["male", "female"]:
            img_dir = os.path.join(root_dir, gender, "images")
            prompt_dir = os.path.join(root_dir, gender, "prompts")
            if not os.path.exists(img_dir): continue

            # NUMERICAL SORTING
            files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")], 
                           key=lambda x: int(os.path.splitext(x)[0]))
            
            prefix = "mh" if gender == "male" else "fh"
            for i, fname in enumerate(files):
                base = os.path.splitext(fname)[0]
                self.all_samples.append({
                    "src_img": os.path.join(img_dir, fname),
                    "src_txt": os.path.join(prompt_dir, f"{base}.txt"),
                    "new_base": f"{prefix}_{i+1:06d}",
                    "gender": gender,
                    "source_id": base
                })

        # SEQUENTIAL SHARDING (Contiguous blocks per GPU)
        n = len(self.all_samples)
        per_gpu = n // world_size
        start = rank * per_gpu
        end = (rank + 1) * per_gpu if rank != world_size - 1 else n
        self.my_samples = self.all_samples[start:end]
        
        if rank == 0 or rank == world_size - 1:
            log(f"Global Total: {n} | Shard Range: {start} to {end} (Local Total: {len(self.my_samples)})")

    def __len__(self):
        return len(self.my_samples)

    def __getitem__(self, idx):
        s = self.my_samples[idx]
        with open(s["src_txt"], "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        
        return {
            "prompt": prompt,
            "sample": s
        }

def get_collate_fn(processor):
    def collate_fn(items):
        encoded = []
        for it in items:
            user_msg = f"Target ID: {it['sample']['source_id']}\nTarget Filename: {it['sample']['new_base']}.png\nGender: {it['sample']['gender']}\nPrompt: {it['prompt']}"
            msgs = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "text", "text": user_msg}]}
            ]
            # applying chat template for text-only
            enc = processor.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            )
            encoded.append(enc)
        
        # Padding
        max_l = max(e["input_ids"].shape[-1] for e in encoded)
        pad_id = processor.tokenizer.pad_token_id or 0
        
        def pad(t, f, m):
            g = m - t.shape[-1]
            if g <= 0: return t
            return torch.cat([torch.full((1, g), f, dtype=t.dtype), t], dim=-1)

        return {
            "input_ids": torch.cat([pad(e["input_ids"], pad_id, max_l) for e in encoded]),
            "attention_mask": torch.cat([pad(e["attention_mask"], 0, max_l) for e in encoded]),
            "meta": [it["sample"] for it in items]
        }
    return collate_fn

class AsyncWriter:
    def __init__(self, target_root):
        self.target_root = target_root
        self.q = queue.Queue(maxsize=512)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        # Open per-gender jsonl handles
        handles = {}
        try:
            while True:
                item = self.q.get()
                if item is None: break
                
                s = item["sample"]
                res_json = item["json"]
                gender = s["gender"]
                
                # 1. Create Directories
                out_img_dir = os.path.join(self.target_root, gender, "images")
                out_txt_dir = os.path.join(self.target_root, gender, "prompts")
                os.makedirs(out_img_dir, exist_ok=True)
                os.makedirs(out_txt_dir, exist_ok=True)

                # 2. Copy/Rename Image (Source might be on slow storage, but shutil is fine here)
                target_img = os.path.join(out_img_dir, f"{s['new_base']}.png")
                shutil.copy2(s["src_img"], target_img)
                
                # 3. Save Txt Prompt
                target_txt = os.path.join(out_txt_dir, f"{s['new_base']}.txt")
                with open(target_txt, "w", encoding="utf-8") as f:
                    f.write(res_json.get("prompt", ""))

                # 4. Save to JSONL
                key = f"{gender}_gpu{RANK}"
                if key not in handles:
                    p = os.path.join(self.target_root, f"{gender}_gpu{RANK}.jsonl")
                    handles[key] = open(p, "a", encoding="utf-8", buffering=1)
                
                handles[key].write(json.dumps(res_json, ensure_ascii=False) + "\n")
                
                self.q.task_done()
        finally:
            for h in handles.values(): h.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    # With --gpus-per-task=1, Slurm masks other GPUs. 
    # Each task sees its assigned GPU as device 0.
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    log(f"Loading processor and model (Text-only mode) on {device}...")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    
    # Still loading the full model, but we won't pass pixels
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True
    ).to(device).eval()

    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    dist.barrier()

    dataset = HumansDataset(args.root_dir, RANK, WORLD_SIZE)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=get_collate_fn(processor), 
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2
    )

    writer = AsyncWriter(args.target_dir)

    log("Starting text-only inference...")
    start_time = time.time()
    
    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                do_sample=False
            )
        
        generated_ids = out[:, input_ids.shape[1]:]
        texts = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for i, text in enumerate(texts):
            try:
                clean_text = text.strip()
                if "```json" in clean_text:
                    clean_text = clean_text.split("```json")[-1].split("```")[0].strip()
                res_json = json.loads(clean_text)
                
                # Enqueue for async saving
                writer.q.put({"sample": batch["meta"][i], "json": res_json})
            except Exception as e:
                log(f"Error parsing JSON for {batch['meta'][i]['new_base']}: {e}")

        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (batch_idx + 1) * args.batch_size / elapsed
            log(f"Batch {batch_idx+1} | Speed: {rate:.2f} samples/sec | Queue: {writer.q.qsize()}")

    log("Inference complete. Waiting for writer...")
    writer.q.put(None)
    writer.thread.join()
    log("All files saved successfully.")

if __name__ == "__main__":
    main()
