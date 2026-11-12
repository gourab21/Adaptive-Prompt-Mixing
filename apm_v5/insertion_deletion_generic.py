"""
Insertion / Deletion scores for APM (DINOv2 ViT-B/14) — generalized over datasets.

Same logic as insertion_deletion.py (CUB-only), parameterized by --dataset so it
also runs on dog / pet / medleaf. See that file's docstring for the full
explanation of how the saliency map is read.

Run from the repo root (apm/):
  python insertion_deletion_generic.py --dataset dog     --checkpoint <path to dog model.pt>
  python insertion_deletion_generic.py --dataset pet     --checkpoint <path to pet model.pt>
  python insertion_deletion_generic.py --dataset medleaf --checkpoint <path to medleaf model.pt>
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_trapz = getattr(np, "trapezoid", None) or np.trapz

IMAGE_SIZE = 224
PATCH_SIZE = 14
GRID_N = IMAGE_SIZE // PATCH_SIZE   # 16 x 16 = 256 patches

INCEPTION_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
INCEPTION_STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)

DATASETS = {
    "cub":     dict(class_num=200, data="cub",     data_path="./data/images/cub",
                     test_annotation="data/annotations/cub/test.txt",
                     module="data.dataset.cub", cls="CubDataset"),
    "dog":     dict(class_num=120, data="dog",     data_path="./data/images/dog",
                     test_annotation="data/annotations/dog/test.txt",
                     module="data.dataset.dog", cls="DogDataset"),
    "pet":     dict(class_num=37,  data="pet",     data_path="./data/images/pet",
                     test_annotation="data/annotations/pet/test.txt",
                     module="data.dataset.pet", cls="PetDataset"),
    "medleaf": dict(class_num=30,  data="medleaf-standard", data_path="./data/images/MedLeaf",
                     test_annotation="data/annotations/medleaf/test.txt",
                     module="data.dataset.medleaf", cls="MedLeafDataset"),
}


# ---------------------------------------------------------------------------
# Model + data
# ---------------------------------------------------------------------------

class _Params:
    pass


def build_params(ds_cfg, data_path):
    p = _Params()
    p.data = ds_cfg["data"]
    p.data_path = data_path
    p.crop_size = IMAGE_SIZE
    p.class_num = ds_cfg["class_num"]
    p.vpt_num = ds_cfg["class_num"]
    p.vpt_mode = None
    p.vpt_layer = None
    p.vpt_dropout = 0.1
    p.drop_path_rate = 0.1
    p.vis_attn = False
    p.debug = False
    p.distributed = False
    p.gpu_num = 1
    p.full = False
    p.model = "dinov2"
    p.pretrained_weights = "vit_base_patch14_dinov2"
    p.train_type = "apm"
    return p


def load_model_and_data(args):
    from experiment.build_model_apm import get_model_apm
    ds_cfg = DATASETS[args.dataset]
    params = build_params(ds_cfg, args.data_path)
    model, _, _ = get_model_apm(params, visualize=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    state = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load] WARNING missing keys ({len(missing)}): {missing[:8]}")
    if unexpected:
        print(f"[load] WARNING unexpected keys ({len(unexpected)}): {unexpected[:8]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    import importlib
    mod = importlib.import_module(ds_cfg["module"])
    dataset_cls = getattr(mod, ds_cfg["cls"])
    transform_raw = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor()])
    dataset = dataset_cls(root=args.data_path, data_list=args.test_annotation, transform=transform_raw)
    return model, dataset, device


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------

def normalize(img01):
    return (img01 - INCEPTION_MEAN.to(img01.device)) / INCEPTION_STD.to(img01.device)


@contextmanager
def vis_attn_mode(model, enabled):
    prev = model.params.vis_attn
    model.params.vis_attn = enabled
    try:
        yield
    finally:
        model.params.vis_attn = prev


@torch.no_grad()
def class_logits(model, x_norm):
    with vis_attn_mode(model, False):
        out, _ = model(x_norm)
    return out.squeeze(-1)


@torch.no_grad()
def get_patch_importance(model, x_norm, target_idx):
    with vis_attn_mode(model, True):
        _, attn_map = model(x_norm, blur_head_lst=[], target_cls=int(target_idx))
    ############# Added module (APM v5) #############
    # Prefer the decoupled static-prompt saliency map (trained to be sharp)
    # over the mixed-prompt classification attention, when the model exposes
    # one (v5 APM). Falls back to the mixed map for other models/versions.
    static_map = getattr(model, '_last_attn_map_static', None)
    if static_map is not None:
        attn_map = static_map
    ############# Added module end #############
    if attn_map is None:
        raise RuntimeError("attn_map is None; the model did not return attention.")
    K = model.params.vpt_num
    row = attn_map[0, :, int(target_idx), (K + 1):]
    return row.mean(dim=0).detach().float().cpu().numpy()


@torch.no_grad()
def target_probs(model, imgs01, target_idx, chunk):
    probs = []
    for i in range(0, imgs01.shape[0], chunk):
        x = normalize(imgs01[i:i + chunk])
        probs.append(F.softmax(class_logits(model, x), dim=-1)[:, target_idx].float().cpu())
    return torch.cat(probs).numpy()


# ---------------------------------------------------------------------------
# Insertion / deletion
# ---------------------------------------------------------------------------

def make_baseline(img01, kind):
    if kind == "gray":
        return torch.full_like(img01, 0.5)
    if kind == "black":
        return torch.zeros_like(img01)
    return TF.gaussian_blur(img01, kernel_size=[51, 51], sigma=[10.0, 10.0])


def build_masks(order, step_fraction, device):
    n = GRID_N * GRID_N
    step = max(1, int(round(n * step_fraction)))
    n_steps = (n + step - 1) // step
    ends = [0] + [min(s * step, n) for s in range(1, n_steps + 1)]

    rank = torch.empty(n, dtype=torch.long)
    rank[torch.as_tensor(order.copy())] = torch.arange(n)
    touched = (rank.view(1, -1) < torch.tensor(ends).view(-1, 1)).float()
    masks = touched.view(-1, 1, GRID_N, GRID_N).to(device)
    masks = masks.repeat_interleave(PATCH_SIZE, dim=2).repeat_interleave(PATCH_SIZE, dim=3)
    return masks, np.array(ends, dtype=np.float64) / n


def curve(model, img01, target_idx, masks, fractions, mode, args):
    base = make_baseline(img01, args.baseline)
    if mode == "deletion":
        stack = img01 * (1 - masks) + base * masks
    else:
        stack = img01 * masks + base * (1 - masks)
    probs = target_probs(model, stack, target_idx, args.eval_batch)
    return float(_trapz(probs, fractions)), probs


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

@torch.no_grad()
def sanity_check(model, dataset, device, args):
    random.seed(args.seed)
    idxs = random.sample(range(len(dataset)), min(args.sanity_check, len(dataset)))
    correct = 0
    for i in range(0, len(idxs), args.eval_batch):
        batch = [dataset[j] for j in idxs[i:i + args.eval_batch]]
        x = torch.stack([b[0] for b in batch]).to(device)
        y = torch.tensor([b[1] for b in batch], device=device)
        correct += int((class_logits(model, normalize(x)).argmax(dim=-1) == y).sum().item())
    acc = 100.0 * correct / len(idxs)
    print(f"[sanity_check] APM DINOv2 ({args.dataset}) top-1 over {len(idxs)} images = {acc:.2f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_path", default=None)
    p.add_argument("--test_annotation", default=None)
    p.add_argument("--n_images", type=int, default=0, help="0 = whole test set")
    p.add_argument("--step_fraction", type=float, default=0.05)
    p.add_argument("--eval_batch", type=int, default=32)
    p.add_argument("--baseline", default="gray", choices=["gray", "blur", "black"])
    p.add_argument("--target", default="gt", choices=["gt", "pred"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="./insertion_deletion_out")
    p.add_argument("--sanity_check", type=int, default=0)
    args = p.parse_args()

    ds_cfg = DATASETS[args.dataset]
    if args.data_path is None:
        args.data_path = ds_cfg["data_path"]
    if args.test_annotation is None:
        args.test_annotation = ds_cfg["test_annotation"]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    model, dataset, device = load_model_and_data(args)

    if args.sanity_check:
        sanity_check(model, dataset, device, args)
        return

    indices = list(range(len(dataset)))
    if args.n_images and args.n_images < len(indices):
        indices = random.Random(args.seed).sample(indices, args.n_images)

    rows, del_curves, ins_curves = [], [], []
    t0 = time.time()

    for n, idx in enumerate(indices, 1):
        image, label = dataset[idx]
        img01 = image.unsqueeze(0).to(device)
        x_norm = normalize(img01)

        pred = int(class_logits(model, x_norm).argmax(dim=-1).item())
        target = int(label) if args.target == "gt" else pred

        importance = get_patch_importance(model, x_norm, target)
        if importance.shape[0] != GRID_N * GRID_N:
            raise RuntimeError(f"Got {importance.shape[0]} patch scores, expected {GRID_N * GRID_N}.")
        order = np.argsort(-importance)

        masks, fractions = build_masks(order, args.step_fraction, device)
        del_auc, del_p = curve(model, img01, target, masks, fractions, "deletion", args)
        ins_auc, ins_p = curve(model, img01, target, masks, fractions, "insertion", args)

        grid = np.linspace(0, 1, 21)
        del_curves.append(np.interp(grid, fractions, del_p))
        ins_curves.append(np.interp(grid, fractions, ins_p))
        rows.append([idx, int(label), pred, int(pred == int(label)), del_auc, ins_auc])

        if n % 50 == 0 or n == len(indices):
            arr = np.array(rows, dtype=np.float64)
            eta = (time.time() - t0) / n * (len(indices) - n) / 60
            print(f"{n}/{len(indices)}  Insertion={arr[:, 5].mean():.4f}  "
                  f"Deletion={arr[:, 4].mean():.4f}  (~{eta:.1f} min left)")

    arr = np.array(rows, dtype=np.float64)
    ok = arr[:, 3] == 1
    n_img = len(rows)
    sem = lambda v: float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    summary = {
        "method": "APM", "backbone": "DINOv2 ViT-B/14", "dataset": args.dataset,
        "checkpoint": args.checkpoint, "n_images": n_img,
        "target": args.target, "baseline": args.baseline, "step_fraction": args.step_fraction,
        "top1_on_sample": float(100.0 * ok.mean()),
        "insertion": float(arr[:, 5].mean()), "insertion_sem": sem(arr[:, 5]),
        "deletion": float(arr[:, 4].mean()), "deletion_sem": sem(arr[:, 4]),
        "insertion_correct_only": float(arr[ok, 5].mean()) if ok.any() else None,
        "deletion_correct_only": float(arr[ok, 4].mean()) if ok.any() else None,
    }

    tag = f"apm_dinov2_{args.dataset}"
    with open(os.path.join(args.out_dir, f"{tag}_per_image.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test_index", "label", "pred", "correct", "deletion_auc", "insertion_auc"])
        w.writerows(rows)
    with open(os.path.join(args.out_dir, f"{tag}_mean_curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fraction"] + [f"{x:.2f}" for x in np.linspace(0, 1, 21)])
        w.writerow(["deletion_prob"] + list(np.mean(del_curves, axis=0)))
        w.writerow(["insertion_prob"] + list(np.mean(ins_curves, axis=0)))
    with open(os.path.join(args.out_dir, f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"=== APM (DINOv2) on {args.dataset}, n={n_img} images ===")
    print(f"Insertion (higher is better): {summary['insertion']:.4f}   (s.e.m. {summary['insertion_sem']:.4f})")
    print(f"Deletion  (lower is better):  {summary['deletion']:.4f}   (s.e.m. {summary['deletion_sem']:.4f})")
    print(f"Top-1 on this sample: {summary['top1_on_sample']:.2f}%")
    print(f"Saved to {args.out_dir}/{tag}_*.csv / .json")


if __name__ == "__main__":
    main()
