"""
Generate class-specific attention heatmaps on CUB-200 test images,
in the visual style of Prompt-CAM's Figure 4 (top attention heads
overlaid on the input image as a jet-colormap heatmap).

Works for BOTH methods via --method {prompt_cam,apm}; run it once per
method so you end up with a matched set of images to place side by
side for the qualitative comparison figure.

Every output is saved as its OWN file, and each image gets its own
subfolder, so you can freely arrange them into a grid in whatever
layout you want for the paper.

For each selected test image, this script creates:
  <outdir>/<idx>_<class>/input.png             -- the plain input image
  <outdir>/<idx>_<class>/head01.png ... head12.png
                                                -- ALL 12 attention heads,
                                                   each overlaid on the
                                                   input as its own file
  <outdir>/<idx>_<class>/avgheads.png          -- all 12 heads averaged
  <outdir>/<idx>_<class>/panel.png             -- all 12 heads laid out
                                                   in a 3-row x 4-column
                                                   grid, purely for a
                                                   quick look; delete it
                                                   if you only want the
                                                   individual files.

Usage:
  CUDA_VISIBLE_DEVICES=0 python make_heatmaps_cub.py --method prompt_cam --num_images 8
  CUDA_VISIBLE_DEVICES=0 python make_heatmaps_cub.py --method apm        --num_images 8
"""

import os
import sys
import argparse
import inspect
import yaml
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ──────────────────────────────────────────────────────────────────────
# Paths  (edit these three if your layout differs)
# ──────────────────────────────────────────────────────────────────────
PROMPTCAM_ROOT = "/home/astra/Desktop/XAI/APM_Final/prompt_cam"
APM_ROOT       = "/home/astra/Desktop/XAI/APM_Final/apm"

PROMPTCAM_CKPT = (PROMPTCAM_ROOT +
    "/output/vit_base_patch14_dinov2/cub/pcam_prompt_cam_200_sgd"
    "/cub/2026-10-30-13:48/model.pt")
APM_CKPT = (APM_ROOT +
    "/output/vit_base_patch14_dinov2/cub/linear_sgd/cub/2026-10-31-08:22/model.pt")

PROMPTCAM_CONFIG = PROMPTCAM_ROOT + "/experiment/config/prompt_cam/dinov2/cub/args.yaml"
APM_CONFIG       = APM_ROOT + "/experiment/config/apm/dinov2/cub/args.yaml"


def load_params_from_yaml(config_path, n_classes, vis_attn=True):
    """Load the exact config used for training, as a SimpleNamespace, so we
    never have to guess which fields get_model/get_model_apm need. We only
    override vis_attn (to get attention maps back) and make sure class_num
    is set."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["vis_attn"] = vis_attn
    cfg.setdefault("class_num", n_classes)
    cfg.setdefault("debug", False)
    cfg.setdefault("distributed", False)
    cfg.setdefault("local_rank", 0)
    cfg.setdefault("output_dir", "./output_heatmap_tmp")
    return SimpleNamespace(**cfg)

VPT_NUM    = 200       # K prompts = number of CUB classes
GRID_N     = 16        # 224 / 14
N_PATCHES  = GRID_N * GRID_N

# CubDataset's own get_cub() applies IMAGENET_INCEPTION_MEAN/STD internally,
# which is (0.5, 0.5, 0.5) for both -- we only need this for denormalizing
# back to a displayable image, not for building the transform ourselves.
mean = std = [0.5, 0.5, 0.5]


def unwrap_model(ret):
    """get_model/get_model_apm may return either the nn.Module directly or
    a tuple like (model, trainable_param_names) / (model, optimizer_cfg).
    Pull out whichever element is actually the model."""
    if hasattr(ret, "load_state_dict"):
        return ret
    if isinstance(ret, (tuple, list)):
        for item in ret:
            if hasattr(item, "load_state_dict"):
                return item
        raise RuntimeError(
            f"get_model(...) returned a tuple/list of length {len(ret)} but "
            f"none of its elements look like an nn.Module (no load_state_dict "
            f"found). Got types: {[type(x) for x in ret]}. "
            f"Print the return value in build_model.py / build_model_apm.py's "
            f"get_model / get_model_apm to see what it actually returns, then "
            f"tell me and I'll fix unwrap_model() to match exactly."
        )
    raise RuntimeError(f"Unexpected return type from get_model(...): {type(ret)}")


def denormalize(img_tensor):
    """img_tensor: (3,224,224) normalized -> numpy HWC in [0,1]."""
    img = img_tensor.clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return img


def build_model_promptcam(device):
    sys.path.insert(0, PROMPTCAM_ROOT)
    from experiment.build_model import get_model

    params = load_params_from_yaml(PROMPTCAM_CONFIG, VPT_NUM, vis_attn=True)
    model  = unwrap_model(get_model(params, VPT_NUM))
    ckpt   = torch.load(PROMPTCAM_CKPT, map_location=device)
    state  = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, params


def build_model_apm(device):
    sys.path.insert(0, APM_ROOT)
    from experiment.build_model_apm import get_model_apm

    params = load_params_from_yaml(APM_CONFIG, VPT_NUM, vis_attn=True)
    model  = unwrap_model(get_model_apm(params, VPT_NUM))
    ckpt   = torch.load(APM_CKPT, map_location=device)
    state  = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, params


def forward(model, params, method, img_tensor, device):
    """Return (logits_1d, attn_map) with attn_map shape (num_heads, M).

    The real model.forward signature (confirmed from a live traceback) is
    forward(self, x, blur_head_lst=[], target_cls=-1) -> (logits, attn_map).
    It does not take params at all, and always returns the attention map
    (forward_features/block.forward compute and return it unconditionally),
    so no vis_attn flag is needed on the call itself.
    """
    inp = img_tensor.unsqueeze(0).to(device)
    try:
        with torch.no_grad():
            out = model(inp)
    except TypeError as e:
        sig = inspect.signature(model.forward)
        raise RuntimeError(
            f"model(inp) call failed: {e}\n"
            f"Actual model.forward signature is: {sig}\n"
            f"Tell me this signature and I will fix the forward() call to match."
        ) from e

    if not isinstance(out, (tuple, list)) or len(out) < 2 or out[1] is None:
        raise RuntimeError(
            f"Expected (logits, attn_map) from model(inp), but got: "
            f"{type(out)} "
            f"{'of length ' + str(len(out)) if isinstance(out, (tuple, list)) else ''}. "
            f"Tell me the model's forward() source and I will fix this."
        )

    logits, attn_map = out[0], out[1]
    if method == "prompt_cam":
        logits = logits.squeeze(-1).squeeze(0)      # (B,K,1) -> (K,)
    else:  # apm
        logits = logits.squeeze(0)                  # (B,K) -> (K,)
    return logits, attn_map


def get_class_attention(attn_map, class_idx):
    """attn_map: (1, num_heads, seq, seq) -> (num_heads, N_PATCHES) for the
    given class prompt's row, patch columns only."""
    heads = attn_map[0, :, class_idx, (VPT_NUM + 1):]    # (num_heads, N_PATCHES)
    return heads.cpu().numpy()


def overlay_heatmap(base_img, attn_1d, grid_n=GRID_N, alpha=0.55):
    """base_img: HWC float [0,1]. attn_1d: (grid_n*grid_n,) attention weights.
    Returns an HWC uint8 image with the jet heatmap overlaid."""
    H, W = base_img.shape[:2]
    heat = attn_1d.reshape(grid_n, grid_n)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat_t = torch.tensor(heat, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    heat_up = F.interpolate(heat_t, size=(H, W), mode="bilinear",
                            align_corners=False).squeeze().numpy()
    jet = cm.get_cmap("jet")(heat_up)[:, :, :3]          # HWC in [0,1]
    blended = (1 - alpha) * base_img + alpha * jet
    blended = np.clip(blended, 0, 1)
    return (blended * 255).astype(np.uint8)


def save_png(arr_uint8_or_float, path):
    if arr_uint8_or_float.dtype != np.uint8:
        arr_uint8_or_float = (np.clip(arr_uint8_or_float, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr_uint8_or_float).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["prompt_cam", "apm"], required=True)
    ap.add_argument("--num_images", type=int, default=8,
                    help="How many correctly classified test images to visualize.")
    ap.add_argument("--num_heads", type=int, default=12,
                    help="Number of attention heads in the model (ViT-B = 12). "
                         "All of them are visualized, no top-K filtering, "
                         "since picking by peak attention value tends to "
                         "surface DINOv2's known high-norm artifact tokens "
                         "instead of genuinely discriminative heads.")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Method: {args.method}")

    out_dir = args.out_dir or f"./heatmaps_{args.method}_cub"
    os.makedirs(out_dir, exist_ok=True)

    if args.method == "prompt_cam":
        sys.path.insert(0, PROMPTCAM_ROOT)
        model, params = build_model_promptcam(device)
    else:
        sys.path.insert(0, APM_ROOT)
        model, params = build_model_apm(device)

    # get_cub() is the dataset factory your training pipeline itself uses:
    # it builds (or reuses) data/annotations/cub/test.txt and returns a
    # CubDataset (an ImageFolder subclass) already wrapped with the correct
    # val-time transform (resize/crop/normalize matching training). Using
    # this instead of constructing CubDataset by hand guarantees the exact
    # same preprocessing and label ordering the model was trained with.
    from data.dataset.cub import get_cub
    try:
        test_ds = get_cub(params, mode="test")
    except Exception as e:
        raise RuntimeError(
            f"get_cub(params, mode='test') failed: {e}\n"
            f"params.data_path={getattr(params, 'data_path', None)!r}, "
            f"params.data={getattr(params, 'data', None)!r}\n"
            f"Make sure you're running this script from inside "
            f"{PROMPTCAM_ROOT if args.method == 'prompt_cam' else APM_ROOT} "
            f"(relative annotation/data paths are resolved from the cwd, "
            f"same as during training)."
        ) from e

    # try to recover human-readable class names if the dataset exposes them
    class_names = getattr(test_ds, "classes", None) or getattr(test_ds, "class_names", None)

    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(test_ds))

    saved = 0
    for idx in order:
        img, label = test_ds[idx]
        label = int(label)
        logits, attn_map = forward(model, params, args.method, img, device)
        pred = int(logits.argmax().item())

        if pred != label:
            continue   # only visualize correctly classified examples, like the paper

        base_img = denormalize(img)
        class_name = (class_names[label] if class_names is not None
                      else f"class{label:03d}")
        safe_name = str(class_name).replace(" ", "_").replace("/", "-")
        tag = f"{saved:02d}_{safe_name}"

        # each image gets its own subfolder
        img_dir = os.path.join(out_dir, tag)
        os.makedirs(img_dir, exist_ok=True)

        # 1. plain input image
        save_png(base_img, os.path.join(img_dir, "input.png"))

        # 2. every head, overlaid, saved as its own file -- no top-K
        #    filtering, since ranking by peak attention value tends to
        #    surface DINOv2's known high-norm artifact/register tokens
        #    rather than the heads that are actually doing useful work.
        heads = get_class_attention(attn_map, label)          # (num_heads, N_PATCHES)
        n_heads = heads.shape[0]

        for h in range(n_heads):
            overlay = overlay_heatmap(base_img, heads[h])
            save_png(overlay, os.path.join(img_dir, f"head{h+1:02d}.png"))

        # 3. bonus: all heads averaged together
        avg_overlay = overlay_heatmap(base_img, heads.mean(axis=0))
        save_png(avg_overlay, os.path.join(img_dir, "avgheads.png"))

        # 4. quick-look combined panel: 3 rows x 4 columns, all 12 heads
        n_rows, n_cols = 3, 4
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows))
        axes = axes.flatten()
        for h in range(n_heads):
            overlay = overlay_heatmap(base_img, heads[h])
            axes[h].imshow(overlay)
            axes[h].set_title(f"Head {h+1}")
            axes[h].axis("off")
        for ax in axes[n_heads:]:
            ax.axis("off")
        fig.suptitle(f"{class_name}  (pred correct)  -- all {n_heads} heads",
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(img_dir, "panel.png"), dpi=150)
        plt.close(fig)

        print(f"[{saved+1}/{args.num_images}] saved: {tag}/ "
              f"({n_heads} heads, true={class_name}, pred_correct=True)")
        saved += 1
        if saved >= args.num_images:
            break

    print(f"\nDone. {saved} correctly classified images visualized.")
    print(f"Output directory: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
