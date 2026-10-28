"""
inference.py — Virtual try-on for one person / garment pair with a checkpoint from train.py.

Uses the same model (SDModel) and denoising loop (run_full_inference) as training and
evaluation, so results match what the model saw during validation.

Usage
-----
python inference.py \
    --checkpoint experiments/<run_name>/checkpoints/ckpt_final.pt \
    --person     person.png \
    --cloth      cloth.png \
    --output     result.png

Writes the try-on to --output and, with --save_grid, a side-by-side
person | garment | try-on image next to it (<output>_grid.png).
Use --ootd for checkpoints trained with train.py --ootd (garment-only conditioning).
"""

import argparse
import os
import sys

import torch
from PIL import Image
from torchvision import transforms

from model import SDModel
from utils import decode_latents, run_full_inference

DEFAULT_IMAGE_SIZE = 512


def load_image(path: str, size: int) -> torch.Tensor:
    """Load an RGB image as a [1, 3, H, W] tensor in [-1, 1]."""
    img = Image.open(path).convert("RGB")
    if size > 0:
        img = transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)(img)
    else:
        # Native resolution: dimensions must be multiples of 8 for the VAE
        w, h = img.size
        img = transforms.Resize(((h // 8) * 8, (w // 8) * 8),
                                interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)(img)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return tf(img).unsqueeze(0)


def to_pil(t: torch.Tensor) -> Image.Image:
    """[1, 3, H, W] tensor in [0, 1] -> PIL image."""
    arr = (t[0].permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


def load_checkpoint(unet, path: str) -> None:
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("unet_state_dict", "model_state_dict", "state_dict"):
        if key in ckpt:
            state_dict = ckpt[key]
            break
    else:
        state_dict = ckpt
    # Strip DDP ("module.") and torch.compile ("_orig_mod.") prefixes
    clean = {}
    for k, v in state_dict.items():
        for prefix in ("module.", "_orig_mod."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        clean[k] = v
    missing, unexpected = unet.load_state_dict(clean, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys, e.g. {missing[:3]}")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    print(f"  loaded (step {ckpt.get('step', '?') if isinstance(ckpt, dict) else '?'})")


@torch.no_grad()
def try_on(model: SDModel, person: torch.Tensor, cloth: torch.Tensor, steps: int, ootd: bool) -> torch.Tensor:
    """Return the try-on image as a [1, 3, H, W] tensor in [0, 1]."""
    # Condition: person and garment side by side (or garment only for --ootd), encoded by the VAE
    cond_input = cloth if ootd else torch.cat([person, cloth], dim=3)
    cond_latents = model.vae.encode(cond_input).latent_dist.sample() * 0.18215

    latents = run_full_inference(model, cond_latents, num_inference_steps=steps)
    decoded = decode_latents(model.vae, latents, decode_batch_size=1, vae_fp16=(latents.device.type == "cuda"))

    # The left half of the side-by-side output is the try-on
    return decoded if ootd else decoded[:, :, :, : cloth.shape[-1]]


def parse_args():
    p = argparse.ArgumentParser(description="CURVTON virtual try-on inference")
    p.add_argument("--checkpoint", required=True, help="Checkpoint .pt written by train.py")
    p.add_argument("--person", required=True, help="Person image")
    p.add_argument("--cloth", required=True, help="Garment image")
    p.add_argument("--output", default="tryon_result.png", help="Where to save the try-on")
    p.add_argument("--steps", type=int, default=50, help="Denoising steps")
    p.add_argument("--size", type=int, default=DEFAULT_IMAGE_SIZE,
                   help="Square resize; <=0 keeps the native resolution")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--device", default="cuda")
    p.add_argument("--ootd", action="store_true", help="Checkpoint was trained with garment-only conditioning")
    p.add_argument("--save_grid", action="store_true", help="Also save person | garment | try-on side by side")
    return p.parse_args()


def main():
    args = parse_args()
    for flag, path in (("--checkpoint", args.checkpoint), ("--person", args.person), ("--cloth", args.cloth)):
        if not os.path.exists(path):
            sys.exit(f"[ERROR] {flag} path does not exist: {path}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.manual_seed(args.seed)
    print(f"Device: {device} | steps: {args.steps} | seed: {args.seed}")

    model = SDModel().to(device)
    model.unet.eval()
    model.vae.eval()
    load_checkpoint(model.unet, args.checkpoint)

    person = load_image(args.person, args.size).to(device)
    cloth = load_image(args.cloth, args.size).to(device)
    if cloth.shape[-2:] != person.shape[-2:]:
        cloth = torch.nn.functional.interpolate(cloth, size=person.shape[-2:], mode="bilinear", align_corners=False)

    result = try_on(model, person, cloth, args.steps, args.ootd)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    to_pil(result).save(args.output)
    print(f"Saved try-on -> {args.output}")

    if args.save_grid:
        grid = torch.cat([(person / 2 + 0.5).clamp(0, 1), (cloth / 2 + 0.5).clamp(0, 1), result.to(person.dtype)], dim=3)
        base, ext = os.path.splitext(args.output)
        to_pil(grid).save(f"{base}_grid{ext or '.png'}")
        print(f"Saved comparison -> {base}_grid{ext or '.png'}")


if __name__ == "__main__":
    main()
