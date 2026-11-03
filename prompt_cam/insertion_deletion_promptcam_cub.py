"""
Insertion / Deletion faithfulness metric
Method : Prompt-CAM (baseline)
Dataset: CUB-200
Backbone: DINOv2 ViT-B/14  (patch_size=14, grid=16x16=256 patches)
vpt_num : 200  (K prompts)

Attention-map token order: [K prompts | cls | N patches]
  -> patch attention at attn_map[0, :, target, (K+1):]
     averaged over all 12 heads.

Results from previous run: Insertion=0.63, Deletion=0.07, n=5794
"""

import os, sys, math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

PCAM_ROOT  = "/home/astra/Desktop/XAI/APM_Final/prompt_cam"
CKPT_PATH  = (PCAM_ROOT +
              "/output/vit_base_patch14_dinov2/cub"
              "/pcam_prompt_cam_200_sgd/cub/2026-10-30-13:48/model.pt")
DATA_PATH  = "/home/astra/Desktop/XAI/APM_Final/shared/data/images/CUB"

sys.path.insert(0, PCAM_ROOT)

VPT_NUM    = 200
N_CLASSES  = 200
PATCH_SIZE = 14
GRID_N     = 16
N_PATCHES  = GRID_N * GRID_N
N_STEPS    = 16

mean = std = [0.5, 0.5, 0.5]
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

GRAY_VALUE = 0.0


def build_model(ckpt_path, n_classes, vpt_num, device):
    from experiment.build_model import get_model

    class Params:
        model              = "dinov2"
        train_type         = "prompt_cam"
        pretrained_weights = "vit_base_patch14_dinov2"
        drop_path_rate     = 0.0
        vpt_num            = vpt_num
        vpt_layer          = None
        vpt_dropout        = 0.0
        vpt_mode           = None
        full               = False
        gpu_num            = 1
        vis_attn           = True

    params = Params()
    model  = get_model(params, n_classes)
    ckpt   = torch.load(ckpt_path, map_location=device)
    state  = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, params


def get_saliency(model, params, img_tensor, device):
    with torch.no_grad():
        inp    = img_tensor.unsqueeze(0).to(device)
        out    = model(inp, params)
        logits = out[0]          # (1, K, 1)
        logits = logits.squeeze(-1).squeeze(0)   # (K,)
        pred   = logits.argmax().item()

        attn_map = out[1]        # (1, 12, seq, seq)
        sal = attn_map[0, :, pred, (VPT_NUM + 1):].mean(0)   # (N_PATCHES,)
    return sal.cpu().numpy(), pred


def prob_with_mask(model, params, img_tensor, pred, mask_flat, device):
    H = W = GRID_N
    img = img_tensor.clone()
    mask_2d  = torch.tensor(mask_flat.reshape(H, W), dtype=torch.bool)
    mask_img = mask_2d.float().unsqueeze(0).unsqueeze(0)
    mask_img = F.interpolate(mask_img, size=(224, 224),
                             mode="nearest").squeeze()
    img[:, ~mask_img.bool()] = GRAY_VALUE
    with torch.no_grad():
        out    = model(img.unsqueeze(0).to(device), params)
        logits = out[0].squeeze(-1).squeeze(0)   # (K,)
        prob   = F.softmax(logits, dim=0)[pred].item()
    return prob


def insertion_deletion_scores(model, params, img_tensor, device,
                               n_steps=N_STEPS):
    sal, pred = get_saliency(model, params, img_tensor, device)
    order     = np.argsort(sal)[::-1]
    total     = N_PATCHES
    step_size = total // n_steps

    ins_probs, del_probs = [], []
    for k in range(1, n_steps + 1):
        n_reveal = min(k * step_size, total)
        revealed = order[:n_reveal]

        ins_mask = np.zeros(total, dtype=bool);  ins_mask[revealed] = True
        del_mask = np.ones(total, dtype=bool);   del_mask[revealed] = False

        ins_probs.append(prob_with_mask(model, params, img_tensor,
                                        pred, ins_mask, device))
        del_probs.append(prob_with_mask(model, params, img_tensor,
                                        pred, del_mask, device))

    xs = np.linspace(0, 1, n_steps)
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(_trapz(ins_probs, xs)), float(_trapz(del_probs, xs))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, params = build_model(CKPT_PATH, N_CLASSES, VPT_NUM, device)

    from data.dataset.cub import CUBDataset
    test_ds = CUBDataset(root=DATA_PATH, train=False, transform=val_transform)
    loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

    ins_all, del_all = [], []
    for i, (img, label) in enumerate(loader):
        img = img.squeeze(0)
        ins, dlt = insertion_deletion_scores(model, params, img, device)
        ins_all.append(ins)
        del_all.append(dlt)
        if (i + 1) % 200 == 0:
            print(f"[{i+1}/{len(test_ds)}]  "
                  f"ins={np.mean(ins_all):.4f}  del={np.mean(del_all):.4f}")

    print(f"\n=== Prompt-CAM  CUB-200  DINOv2 ===")
    print(f"  Insertion AUC : {np.mean(ins_all):.4f}")
    print(f"  Deletion  AUC : {np.mean(del_all):.4f}")
    print(f"  n samples     : {len(ins_all)}")


if __name__ == "__main__":
    main()
