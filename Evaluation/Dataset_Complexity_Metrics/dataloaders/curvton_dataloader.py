"""
dataloaders/curvton_dataloader.py
===================================
CURVTON Dataset Dataloader

Supports:
- Easy / Medium / Hard difficulty splits
- Male / Female subsets
- Configurable sampling (10%, 20%, ..., 100%)
- Train and Test splits

Directory structure:
    {base_path}/
    ├── easy/
    │   ├── female/
    │   │   ├── cloth_image/
    │   │   ├── initial_person_image/
    │   │   └── tryon_image/
    │   └── male/
    │       ├── cloth_image/
    │       ├── initial_person_image/
    │       └── tryon_image/
    ├── medium/
    │   └── ...
    └── hard/
        └── ...

Usage:
    from dataloaders.curvton_dataloader import CURVTONDataloader
    
    loader = CURVTONDataloader(
        base_path="/path/to/dataset_ultimate",
        difficulty="easy",  # or "medium", "hard", "all"
        gender="all",       # or "male", "female"
        sample_ratio=1.0,   # 0.1, 0.2, ..., 1.0
    )
    
    for person_img, cloth_img, tryon_img, metadata in loader:
        ...
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
from PIL import Image


DifficultyLevel = Literal["easy", "medium", "hard", "all"]
GenderType = Literal["male", "female", "all"]


class CURVTONDataloader:
    """
    CURVTON Dataset Dataloader with difficulty-based splits.
    
    Parameters
    ----------
    base_path : str
        Root directory of CURVTON dataset
    difficulty : str
        One of "easy", "medium", "hard", or "all"
    gender : str
        One of "male", "female", or "all"
    sample_ratio : float
        Fraction of dataset to use (0.1 to 1.0)
    seed : int
        Random seed for reproducible sampling
    return_paths : bool
        If True, returns file paths instead of loaded images
    """
    
    DIFFICULTIES = ["easy", "medium", "hard"]
    GENDERS = ["male", "female"]
    
    # Approximate sample counts per split (from user's structure)
    SAMPLE_COUNTS = {
        "easy": {"female": 33657, "male": 30543},
        "medium": {"female": 37760, "male": 29541},
        "hard": {"female": 32488, "male": 28277},
    }
    
    def __init__(
        self,
        base_path: str,
        difficulty: DifficultyLevel = "all",
        gender: GenderType = "all",
        sample_ratio: float = 1.0,
        seed: int = 42,
        return_paths: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.base_path = Path(base_path)
        self.difficulty = difficulty
        self.gender = gender
        self.sample_ratio = np.clip(sample_ratio, 0.01, 1.0)
        self.seed = seed
        self.return_paths = return_paths
        self.max_samples = max_samples
        
        # Validate base path
        if not self.base_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.base_path}")
        
        # Collect all samples
        self.samples = self._collect_samples()
        
        # Apply sampling
        if self.sample_ratio < 1.0 or self.max_samples is not None:
            self._apply_sampling()
        
        print(f"[CURVTON] Loaded {len(self.samples)} samples "
              f"(difficulty={difficulty}, gender={gender}, ratio={sample_ratio:.0%})")
    
    def _collect_samples(self) -> List[Dict]:
        """Collect all valid triplets (person, cloth, tryon)."""
        samples = []
        
        difficulties = self.DIFFICULTIES if self.difficulty == "all" else [self.difficulty]
        genders = self.GENDERS if self.gender == "all" else [self.gender]
        
        for diff in difficulties:
            diff_path = self.base_path / diff
            if not diff_path.exists():
                print(f"  [CURVTON] Warning: {diff_path} not found, skipping")
                continue
            
            for gen in genders:
                gen_path = diff_path / gen
                if not gen_path.exists():
                    continue
                
                cloth_dir = gen_path / "cloth_image"
                person_dir = gen_path / "initial_person_image"
                tryon_dir = gen_path / "tryon_image"
                
                if not all(d.exists() for d in [cloth_dir, person_dir, tryon_dir]):
                    continue
                
                # Index cloth images
                cloth_files = {f.stem: f for f in cloth_dir.glob("*.png")}
                
                # Index tryon images (they have matching cloth names)
                for tryon_file in tryon_dir.glob("*.png"):
                    tryon_stem = tryon_file.stem
                    
                    # Extract person ID from tryon filename
                    # Format: fh_000001_e01_fc_010632_tracksuit
                    parts = tryon_stem.split("_")
                    if len(parts) >= 3:
                        # Person ID: fh_000001_e01 or mh_000001_e01
                        person_id = "_".join(parts[:3])
                        person_file = person_dir / f"{person_id}.png"
                        
                        if person_file.exists() and tryon_stem in cloth_files:
                            cloth_file = cloth_files[tryon_stem]
                            
                            # Check for metadata JSON
                            json_file = tryon_dir / f"{tryon_stem}.json"
                            metadata = self._load_metadata(json_file) if json_file.exists() else {}
                            
                            samples.append({
                                "person_path": str(person_file),
                                "cloth_path": str(cloth_file),
                                "tryon_path": str(tryon_file),
                                "difficulty": diff,
                                "gender": gen,
                                "person_id": person_id,
                                "cloth_id": tryon_stem,
                                "metadata": metadata,
                            })
        
        return samples
    
    def _load_metadata(self, json_path: Path) -> Dict:
        """Load metadata from JSON file."""
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _apply_sampling(self):
        """Apply random sampling to reduce dataset size."""
        rng = random.Random(self.seed)
        
        n_samples = len(self.samples)
        target_n = int(n_samples * self.sample_ratio)
        
        if self.max_samples is not None:
            target_n = min(target_n, self.max_samples)
        
        if target_n < n_samples:
            self.samples = rng.sample(self.samples, target_n)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self) -> Iterator[Tuple]:
        for sample in self.samples:
            if self.return_paths:
                yield (
                    sample["person_path"],
                    sample["cloth_path"],
                    sample["tryon_path"],
                    sample,
                )
            else:
                try:
                    person_img = Image.open(sample["person_path"]).convert("RGB")
                    cloth_img = Image.open(sample["cloth_path"]).convert("RGB")
                    tryon_img = Image.open(sample["tryon_path"]).convert("RGB")
                    yield (person_img, cloth_img, tryon_img, sample)
                except Exception as e:
                    print(f"  [CURVTON] Error loading {sample['person_id']}: {e}")
                    continue
    
    def __getitem__(self, idx: int) -> Tuple:
        sample = self.samples[idx]
        if self.return_paths:
            return (
                sample["person_path"],
                sample["cloth_path"],
                sample["tryon_path"],
                sample,
            )
        else:
            person_img = Image.open(sample["person_path"]).convert("RGB")
            cloth_img = Image.open(sample["cloth_path"]).convert("RGB")
            tryon_img = Image.open(sample["tryon_path"]).convert("RGB")
            return (person_img, cloth_img, tryon_img, sample)
    
    def get_difficulty_subset(self, difficulty: str) -> "CURVTONDataloader":
        """Return a new loader with only samples from specified difficulty."""
        subset = CURVTONDataloader.__new__(CURVTONDataloader)
        subset.base_path = self.base_path
        subset.difficulty = difficulty
        subset.gender = self.gender
        subset.sample_ratio = self.sample_ratio
        subset.seed = self.seed
        subset.return_paths = self.return_paths
        subset.max_samples = self.max_samples
        subset.samples = [s for s in self.samples if s["difficulty"] == difficulty]
        return subset
    
    def get_stats(self) -> Dict:
        """Return dataset statistics."""
        stats = {
            "total": len(self.samples),
            "by_difficulty": {},
            "by_gender": {},
            "by_difficulty_gender": {},
        }
        
        for diff in self.DIFFICULTIES:
            stats["by_difficulty"][diff] = sum(
                1 for s in self.samples if s["difficulty"] == diff
            )
        
        for gen in self.GENDERS:
            stats["by_gender"][gen] = sum(
                1 for s in self.samples if s["gender"] == gen
            )
        
        for diff in self.DIFFICULTIES:
            for gen in self.GENDERS:
                key = f"{diff}_{gen}"
                stats["by_difficulty_gender"][key] = sum(
                    1 for s in self.samples
                    if s["difficulty"] == diff and s["gender"] == gen
                )
        
        return stats


def create_curvton_loaders(
    base_path: str,
    sample_ratios: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
    seed: int = 42,
) -> Dict[str, Dict[str, CURVTONDataloader]]:
    """
    Create CURVTON dataloaders for all difficulty levels and sample ratios.
    
    Returns
    -------
    Dict with structure:
        {
            "10%": {"easy": loader, "medium": loader, "hard": loader, "all": loader},
            "20%": {...},
            ...
        }
    """
    loaders = {}
    
    for ratio in sample_ratios:
        ratio_key = f"{int(ratio * 100)}%"
        loaders[ratio_key] = {}
        
        for diff in ["easy", "medium", "hard", "all"]:
            loaders[ratio_key][diff] = CURVTONDataloader(
                base_path=base_path,
                difficulty=diff,
                sample_ratio=ratio,
                seed=seed,
                return_paths=True,
            )
    
    return loaders


# ═══════════════════════════════════════════════════════════════════════════════
# CLI for testing
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test CURVTON dataloader")
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--difficulty", type=str, default="all")
    parser.add_argument("--gender", type=str, default="all")
    parser.add_argument("--sample_ratio", type=float, default=0.01)
    args = parser.parse_args()
    
    loader = CURVTONDataloader(
        base_path=args.base_path,
        difficulty=args.difficulty,
        gender=args.gender,
        sample_ratio=args.sample_ratio,
        return_paths=True,
    )
    
    print("\nDataset Statistics:")
    stats = loader.get_stats()
    for key, val in stats.items():
        print(f"  {key}: {val}")
    
    print("\nSample entries:")
    for i, (person, cloth, tryon, meta) in enumerate(loader):
        if i >= 3:
            break
        print(f"  [{i}] Person: {Path(person).name}")
        print(f"       Cloth:  {Path(cloth).name}")
        print(f"       Tryon:  {Path(tryon).name}")
        print(f"       Diff:   {meta['difficulty']}, Gender: {meta['gender']}")
