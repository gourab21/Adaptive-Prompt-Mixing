"""
Adaptive Prompt Mixing (APM)
----------------------------
Replaces Prompt-CAM's static per-class prompts with a dynamic mixture
computed from the input image's CLS token.

At each transformer layer:
  1. A lightweight linear mixer (W_mix: K x D) projects the frozen CLS
     token into K mixing weights via softmax.
  2. The resulting weight vector alpha soft-combines all K class prompt
     embeddings into a single mixed prompt of shape (K, D).
  3. This mixed prompt is injected exactly as Prompt-CAM does, so the
     rest of the forward pass (CAM extraction, head) is unchanged.

Parameter count added vs Prompt-CAM:
  W_mix: K * D  (e.g. CUB: 200 * 768 = 153,600 — < 0.2% of backbone)

APM v2 note: this module (the mixer) is unchanged from v1. The v1->v2 fix lives
in engine/trainer.py and model/vision_transformer.py: v1's attention map was
diffuse enough to lose on deletion AUC despite winning insertion (see
CLAUDE.md open issue #1). v2 adds an entropy penalty, during training only, on
the last-layer attention row from the ground-truth class prompt to the patch
tokens (the same row insertion/deletion reads as the saliency map), to pull
that map back toward fewer, higher-confidence patches.

APM v3 note: this module is still unchanged. v3 extends the v2 trainer-side
fix with two more levers (see engine/trainer.py): (1) entropy_warmup_epochs
ramps the entropy penalty in instead of applying it at full strength from
epoch 0, so it doesn't fight the representation while it's still forming;
(2) a top-k mass penalty directly rewards concentrating the saliency row's
mass into its own top-k patches, which is closer to what deletion AUC
literally measures than entropy alone.
"""

import math
import torch
import torch.nn as nn
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair


class APM(nn.Module):
    """Adaptive Prompt Mixing module."""

    def __init__(self, params, depth, patch_size, embed_dim):
        super().__init__()
        self.params = params
        self.depth = depth
        self.K = params.vpt_num       # number of classes == number of prompts
        self.D = embed_dim

        # --- class prompt embeddings: same shape as Prompt-CAM's VPT ---
        # shape: (num_layers, K, D)
        val = math.sqrt(6. / float(3 * reduce(mul, _pair(patch_size), 1) + embed_dim))
        self.prompt_embeddings = nn.Parameter(
            torch.zeros(depth, self.K, embed_dim))
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        self.prompt_dropout = nn.Dropout(params.vpt_dropout)

        # --- mixing weights: one shared linear layer across all layers ---
        # Maps CLS token (D,) -> mixing weights over K classes
        self.mixer = nn.Linear(embed_dim, self.K, bias=False)
        nn.init.zeros_(self.mixer.weight)   # start near uniform softmax

        # temperature for softmax — learned scalar, init=1
        self.log_temp = nn.Parameter(torch.zeros(1))

    def retrieve_prompt(self, layer_idx, x):
        """
        Args:
            layer_idx: which transformer layer we are at
            x: current sequence tensor of shape (B, N, D)
               token layout before prompt injection: [cls | patch_tokens]
               cls token is x[:, 0, :]

        Returns:
            mixed_prompt: (B, K, D)  — to be prepended to x
        """
        cls_token = x[:, 0, :]                          # (B, D)  frozen by backbone

        # mixing weights
        temp = self.log_temp.exp().clamp(min=0.07)      # temperature in [0.07, inf)
        logits = self.mixer(cls_token) / temp            # (B, K)
        alpha = torch.softmax(logits, dim=-1)            # (B, K)

        # class prompts for this layer
        prompts = self.prompt_dropout(
            self.prompt_embeddings[layer_idx])           # (K, D)

        # weighted combination: alpha (B,K) x prompts (K,D) -> (B, K, D)
        mixed = torch.einsum('bk,kd->bkd', alpha, prompts)

        return mixed, alpha   # return alpha for logging / CAM analysis
