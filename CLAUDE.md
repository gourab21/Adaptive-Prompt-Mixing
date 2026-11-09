# APM Project — Context for Claude Code

**Read this first, in full, before touching any code.** It summarizes everything done on this project so far, in a different chat/session, so that work isn't repeated or contradicted.

> **Note on an older file you may find in this repo or in old notes:** there is an earlier document called `APM_PROJECT_CONTEXT.md` describing a *different, earlier* method also called "APM" ("Adaptive Prompt Mixing", a CLS-token-driven dynamic mixer, tested on 3 datasets, no MedLeaf). **That version was superseded.** The method actually built, trained, evaluated, and written up into a paper is the one described below ("Attentive Prompt Mixer"). If you find both documents, trust this one.

---

## 1. What this project is

Final-year research project extending **Prompt-CAM** (Chowdhury, Paul et al., *CVPR 2025*, [arXiv:2501.09333](https://arxiv.org/abs/2501.09333)) — a method that makes vision transformers interpretable for fine-grained classification by giving each class its own learnable "prompt" token, scored independently.

**Our contribution ("APM" = Attentive Prompt Mixer):** let the K class-specific prompts attend to each other through a small learned mixer before scoring, instead of scoring each independently. Same frozen backbone, same training recipe, only the mixer is new.

Supervisor: Dr. Atanu Mondal.

---

## 2. The method, precisely

**Baseline (Prompt-CAM):**
```
s[c] = wᵗ · z_c            # each class prompt scored independently against one shared vector w
```

**Ours (APM):**
```
Z̃ = Z + softmax(M·Zᵗ)ᵗ·Z                        # cross-prompt mixing
s[c] = (1 / exp(log τ)) · z̃_cᵗ · z̃_c              # scoring on the mixed prompts
```
- `Z` = the K class-specific prompt embeddings (output of the last transformer layer, before scoring)
- `M` = a learned `K × D` mixer matrix (new)
- `log τ` = a learned temperature scalar (new)
- Backbone is **fully frozen**. Only the prompts, `M`, and `log τ` are trained — same as the baseline's trainable set, plus these two.
- **Overhead:** ≈153K extra parameters on CUB-200 (K=200, D=768 → 200×768), vs 86M frozen backbone parameters.

**Backbone used in all reported results:** DINOv2 ViT-B/14 (frozen), patch size 14, 224×224 input → 16×16 = 256 patches.

---

## 3. Datasets and results so far

Four datasets, identical splits/preprocessing for both methods: **CUB-200-2011, Stanford Dogs, Oxford-IIIT Pet, MedLeaf.**

### Classification accuracy (Top-1 / Top-5, %)

| Dataset | Baseline Top-1 | Baseline Top-5 | APM Top-1 | APM Top-5 |
|---|---|---|---|---|
| CUB-200-2011 | 72.23 | 91.23 | **85.11** | **97.08** |
| Stanford Dogs | 80.06 | 96.49 | **84.41** | **98.03** |
| Oxford-IIIT Pet | 92.78 | 99.48 | **93.87** | **99.65** |
| MedLeaf | 99.52 | 100.00 | **100.00** | **100.00** |

APM wins Top-1 on every dataset; the gap is largest on CUB-200 (+12.88 points), the dataset with the most classes and the finest visual distinctions.

### Saliency faithfulness — CUB-200 only (insertion/deletion AUC, RISE protocol — Petsiuk et al. 2018)

| Method | Insertion AUC | Deletion AUC | Insertion (correct only) | Deletion (correct only) |
|---|---|---|---|---|
| Baseline | 0.6315 | 0.0737 | 0.8345 | 0.0931 |
| APM | 0.7919 | 0.1020 | 0.9114 | 0.1142 |

n = 5,794 (full CUB-200 test split). Protocol: ground-truth-class target, gray baseline, step fraction 0.05.

**⚠️ UNRESOLVED ISSUE — read before writing anything about this table:**
- **Insertion: higher is better.** APM wins this cleanly (0.79 vs 0.63).
- **Deletion: lower is better** (standard RISE convention — a good explanation makes the prediction collapse fast when the important patches are removed, so a *small* area under the curve is good). The eval scripts' own docstrings confirm this: *"Insertion (higher is better)... Deletion (lower is better)."*
- **The current draft of `apm_paper.tex` has this backwards** — its table caption says "Higher is better for both metrics" and bolds APM's deletion number as a win. Under the correct convention, **APM's deletion AUC (0.1020) is actually worse than the baseline's (0.0737)** — the baseline wins on deletion.
- **This needs to be fixed in the paper before submission**: either correct the caption/claim to be honest about this (APM wins insertion and loses deletion), or improve the architecture so APM actually wins deletion too (see Section 6, item 1).
- Faithfulness has only been evaluated on **CUB-200**. Dogs, Pet, and MedLeaf have no insertion/deletion numbers yet.

### Training dynamics (from parsed logs)
- **CUB-200:** baseline overfits heavily (99.98% train / 72.23% test). APM generalizes much better (91.08% train / 85.11% test).
- **Stanford Dogs:** baseline suffers a sharp training collapse around epoch 26–33 (validation accuracy briefly falls to ~1%). APM shows no such instability.

### Qualitative (all 12 attention heads, one CUB-200 test image, class 148)
- Baseline: most heads redundantly repeat the same wing/torso region; a vertical streak + 3 corner artifacts appear across nearly all 12 heads (consistent with DINOv2's documented "register token" / high-norm-token phenomenon, Darcet et al. ICLR 2024).
- APM: less artifact contamination, more head diversity — different heads track the head/crown, neck, and body independently.

---

## 4. Repository / codebase layout (as actually used during debugging)

```
/home/astra/Desktop/XAI/APM_Final/
├── prompt_cam/                              # baseline model code
├── apm/                                     # our method's model code
├── experiment/
│   └── config/
│       ├── prompt_cam/dinov2/{cub,dog,pet,medleaf}/args.yaml
│       └── apm/dinov2/{cub,dog,pet,medleaf}/args.yaml
├── data/
│   └── dataset/
│       └── cub.py                           # defines class CubDataset(ImageFolder) + get_cub(params, mode=...)
│                                             # (and equivalent files for dog/pet/medleaf)
└── output/                                  # checkpoints, e.g. .../cub/linear_sgd/cub/<date>/model.pt
```

**Key implementation facts confirmed by direct debugging of this codebase (do not re-guess these):**
- `get_model(params, VPT_NUM)` and `get_model_apm(params, VPT_NUM)` return a **tuple**, not a bare model. Unwrap by finding the element with `load_state_dict`.
- Model `forward` signature is `forward(self, x, blur_head_lst=[], target_cls=-1) -> Tuple[logits, attn_map]`. It **always** returns `(logits, attn_map)`. Do not pass extra positional args — a prior bug came from passing a `params`/`SimpleNamespace` object positionally into `blur_head_lst`.
- Dataset loading goes through the project's own factory functions, e.g. `from data.dataset.cub import get_cub; ds = get_cub(params, mode="test")` — **not** a hand-constructed `CUBDataset(root=..., train=...)`.
- Config loading: load the actual training YAML (`experiment/config/<method>/dinov2/<dataset>/args.yaml`) as a namespace rather than hand-rolling a params object — hand-rolled params objects kept missing attributes (`class_num`, `debug`, etc.) one at a time.
- Attention map shape: `[B, heads, L, L]`, where `L = K (prompts) + 1 (cls) + N (patches)`. The per-class attention row is `attn_map[0, :, class_idx, (K+1):]` → shape `(heads, N)`, `N` = 256 for a 224×224 image at patch size 14.
- Class names are not populated via `.classes` on `CubDataset` (it's populated through a custom annotation-file mechanism), so visualization code falls back to `class{label:03d}`.

**Hardware used for all training/eval runs:** dual Intel Xeon Silver 4310 (24 cores / 48 threads), 125 GiB RAM, NVIDIA RTX A6000 (48 GB), driver 580.178.04, CUDA 13.0, Ubuntu 24.04.5 LTS.

---

## 5. Deliverables already produced (outside the training codebase)

- **`apm_paper.tex`** + **`references.bib`** — full LaTeX research paper (title page, abstract, intro, related work, method, experimental setup, results, discussion, conclusion). Compiles cleanly (pdflatex → bibtex → pdflatex → pdflatex), 12 pages. **Contains the deletion-metric error described in Section 3 above — needs a fix.**
- **`make_heatmaps_cub.py`** — generates per-head attention heatmaps on CUB-200 test images for both methods (all 12 heads, 3×4 panel grid, one subfolder per image).
- **`insertion_deletion_promptcam_cub.py` / `insertion_deletion_apm_cub.py`** (and a more general `insertion_deletion.py` / `insertion_deletion_promptcam.py`) — the faithfulness evaluation scripts that produced the numbers in Section 3. Currently CUB-200 only.
- **TikZ architecture diagrams** for both Prompt-CAM and APM (standalone `.tex`, compiles to PNG), used in both the paper and the slide deck.
- **A 12-slide presentation deck** (cover, intro w/ highlights, research gap, proposed method, both architecture diagrams, implementation + hardware spec table, accuracy results, faithfulness results, training curves, qualitative heatmap comparison, conclusion) — built for a 10-minute recorded video presentation, white background, large Times New Roman text, citations inline on relevant slides.
- **A presentation script** — a first-person speaking script, one section per slide, timed to roughly match a 10-minute assignment brief (intro 1 min, gap 2 min, proposed+architecture 2 min, implementation 2 min, results 2 min, conclusion 1 min).
- **`README.md`** — GitHub-facing project README (overview, both architectures, both results tables, repo structure, setup/usage commands, citations, AI-usage declaration, license placeholder).

---

## 6. Open problems / next steps

1. **Fix the deletion-metric issue in the paper (priority).** Either (a) correct the claim/caption honestly (APM wins insertion, loses deletion — discuss why), or (b) modify the architecture to also win on deletion. The likely cause: the mixer spreads attention mass across more patches (which is *why* insertion improves), so removing the top-ranked patches alone doesn't collapse the prediction as fast. A candidate fix is adding a sparsity/entropy penalty on the mixer's softmax (or a sharper temperature) to pull the attention back toward fewer patches. **This is an architecture/loss change, so it requires retraining**, not just re-evaluating existing checkpoints.
2. **Extend faithfulness evaluation (insertion/deletion) to Dogs, Pet, and MedLeaf** — currently CUB-200 only.
3. **Possible scale-up to 8 datasets** (currently 4) was discussed as a way to turn this into a stronger, more thorough paper — would need new training runs, new accuracy/faithfulness tables, and a base-APM-vs-modified-APM ablation with justification for each architectural change.
4. **Literature check (already done):** searched for prior work combining Prompt-CAM with any cross-prompt mixing mechanism — found none. Prompt-CAM itself (CVPR 2025) and a separate, unrelated method called Finer-CAM ([arXiv:2501.11309](https://arxiv.org/pdf/2501.11309), same research group) exist, but no prior mixer-on-Prompt-CAM work. The idea appears novel.
5. **Publication target (discussed, not finalized):** IEEE Access recommended as the safer/faster option given the paper's empirical-breadth style; ICPR is a stretch option (more prestige, more competitive on novelty); Pattern Recognition Letters or a CVPR/ICCV/WACV workshop track as a middle ground. Leaning toward IEEE Access, especially if the dataset count is expanded to 8 and a careful ablation (base APM vs. sparsity-modified APM) is added.
