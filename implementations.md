# Implementations — Architecture and Results (extracted from training logs)

## 1. Architecture actually used in `apm/`

The code in `apm/model/apm.py` + `apm/model/vision_transformer_apm.py` does **not** implement
cross-prompt attention. What it actually does, at every transformer layer `l`:

1. Take the current CLS token `x_cls (B, D)` flowing through the backbone at layer `l`.
2. `logits = Linear(D -> K)(x_cls) / temp` — a single linear layer (`self.mixer`, no bias),
   shared across all 12 layers, maps the CLS token to K raw mixing scores. `temp = exp(log_temp)`
   is one learned scalar, clamped to `>= 0.07`.
3. `alpha = softmax(logits, dim=-1)` — a `(B, K)` distribution over the K static class-prompt slots.
4. `prompts[l]` — the K per-layer static prompt embeddings (`nn.Parameter` of shape
   `(depth, K, D)`, same shape/init as Prompt-CAM's VPT prompts).
5. `mixed = alpha ⊗ prompts[l]` (`torch.einsum('bk,kd->bkd', alpha, prompts[l])`) — each of the K
   prompt embeddings is **rescaled** by its own scalar weight `alpha[:, k]`. There is no
   prompt-to-prompt mixing (no `prompt_i` ever reads `prompt_j`'s content) — this is purely a
   CLS-token-conditioned gate on the existing K static prompts.
6. `mixed` is prepended to the sequence exactly where Prompt-CAM prepends its static prompts, and
   the rest of the forward pass (self-attention blocks, CAM extraction) is **completely
   unmodified** — `model/attention.py`, `model/block.py`, `model/mlp.py`, `model/patch_embed.py`,
   `model/utils.py` and the parent `model/vision_transformer.py` are byte-identical between
   `apm/` and `prompt_cam/`.
7. Scoring head: unchanged shared `nn.Linear(embed_dim, 1)` applied independently to each of the
   K output prompt tokens, i.e. `s[c] = wᵗ·z_c + b` — identical in form to the Prompt-CAM baseline.

**Trainable parameters added over Prompt-CAM:** the `mixer` linear layer (`K × D`, no bias) and
one scalar `log_temp`. The static `prompt_embeddings (depth, K, D)` and the shared head are
trained in both methods already, so they are not "new" parameters, just re-used.

**Backbone:** DINOv2 ViT-B/14, frozen, `init_values=1e-5`, `img_size=224` (confirmed from
`pretrained_weights='vit_base_patch14_dinov2'` in every run's logged `Namespace`).

Dead code: `apm/model/vpt.py` is an unused leftover, byte-identical to `prompt_cam/model/vpt.py`
— `vision_transformer_apm.py` deletes `self.vpt` right after construction and uses `self.apm`
instead.

**Note:** this is a CLS-token-conditioned softmax gate over static prompts, not the cross-prompt
self-attention mixer (`Z̃ = Z + softmax(M·Zᵗ)ᵗ·Z`, scored by `z̃_cᵗ·z̃_c`) described in
`CLAUDE.md` §2. The numbers below were produced by the architecture described above, as run by
`apm/main.py` with `train_type='apm'`.

---

## 2. Run configuration per dataset (from each log's `Namespace` line)

| Dataset | Method | lr | optimizer | epochs | warmup | batch size | vpt_num (K) | drop_path_rate | vpt_dropout |
|---|---|---|---|---|---|---|---|---|---|
| CUB-200 | Prompt-CAM | 0.001 | sgd | 130 | 20 | 16 | 200 | 0.0 | 0.0 |
| CUB-200 | APM | 0.005 | sgd | 130 | 20 | 16 | 200 | 0.1 | 0.1 |
| Stanford Dogs | Prompt-CAM | 0.005 | sgd | 100 | 20 | 16 | 120 | 0.0 | 0.0 |
| Stanford Dogs | APM | 0.002 | sgd | 100 | 10 | 16 | 120 | 0.1 | 0.1 |
| Oxford-IIIT Pet | Prompt-CAM | 0.005 | sgd | 100 | 20 | 16 | 37 | 0.0 | 0.0 |
| Oxford-IIIT Pet | APM | 0.002 | sgd | 100 | 10 | 16 | 37 | 0.1 | 0.1 |
| MedLeaf | Prompt-CAM | 0.001 | sgd | 130 | 20 | 16 | 30 | 0.0 | 0.0 |
| MedLeaf | APM | 0.005 | sgd | 130 | 20 | 16 | 30 | 0.1 | 0.1 |

---

## 3. Accuracy — final-epoch train vs. test (from the last logged epoch / `Inference (test)` line)

| Dataset | Method | Train Top-1 | Train Top-5 | Test Top-1 | Test Top-5 | Total run time (min) |
|---|---|---|---|---|---|---|
| CUB-200 | Prompt-CAM | 99.98 | 100.00 | 72.23 | 91.23 | 198.73 |
| CUB-200 | APM | 91.08 | 97.29 | **85.11** | **97.08** | 203.28 |
| Stanford Dogs | Prompt-CAM | 99.83 | 100.00 | 80.06 | 96.49 | 316.30 |
| Stanford Dogs | APM | 87.39 | 97.28 | **84.41** | **98.03** | 324.43 |
| Oxford-IIIT Pet | Prompt-CAM | 100.00 | 100.00 | 92.78 | 99.48 | 57.48 |
| Oxford-IIIT Pet | APM | 94.65 | 99.13 | **93.87** | **99.65** | 59.08 |
| MedLeaf | Prompt-CAM | 100.00 | 100.00 | 99.52 | 100.00 | 49.36 |
| MedLeaf | APM | 99.36 | 99.93 | **100.00** | 100.00 | 49.72 |

Prompt-CAM overfits heavily on every dataset (train Top-1 ≈ 100%, large train/test gap). APM's
train Top-1 is consistently 5–13 points lower than its own test Top-1 margin over the baseline,
and its train/test gap is much smaller — i.e. the CLS-conditioned gate acts as a regularizer.

---

## 4. Saliency faithfulness — insertion/deletion AUC (RISE protocol)

Only run for **CUB-200** (`n = 5,794`, full test split, ground-truth target class, gray baseline,
step fraction 0.05). No insertion/deletion runs exist yet for Dogs, Pet, or MedLeaf.

| Method | Top-1 on eval sample | Insertion AUC ↑ | Deletion AUC ↓ | Insertion (correct only) ↑ | Deletion (correct only) ↓ |
|---|---|---|---|---|---|
| Prompt-CAM | 72.23 | 0.6315 | **0.0737** | 0.8345 | **0.0931** |
| APM | 85.11 | **0.7919** | 0.1020 | **0.9114** | 0.1142 |

- Insertion: higher is better — APM wins.
- Deletion: **lower is better** — Prompt-CAM wins here; APM's deletion AUC is worse (0.1020 vs
  0.0737). This is the opposite of what `apm_paper.tex` currently claims — see `CLAUDE.md` §3/§6
  for the open issue.
