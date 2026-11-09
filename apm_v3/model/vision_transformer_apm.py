"""
VisionTransformerAPM
--------------------
Subclass of VisionTransformerPETL that swaps the static VPT module
for the Adaptive Prompt Mixing (APM) module.

Only forward_features, forward_head and the constructor differ from the parent.
Everything else (trainer, data loaders, CAM head) is reused unchanged.
"""

from typing import Tuple, Optional, Callable, Type, Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
from timm.models._registry import register_model
from timm.models._builder import build_model_with_cfg
from timm.models.vision_transformer import checkpoint_filter_fn
from timm.layers import get_norm_layer, get_act_layer

from model.vision_transformer import VisionTransformerPETL, _create_vision_transformer_petl
from model.block import BlockPETL
from model.patch_embed import PatchEmbedPETL
from model.mlp import MlpPETL
from model.apm import APM
from utils.setup_logging import get_logger

logger = get_logger("Prompt_CAM")


class VisionTransformerAPM(VisionTransformerPETL):
    """ViT with Adaptive Prompt Mixing."""

    def __init__(self, *args, **kwargs):
        # Let parent build everything, then replace vpt with apm
        super().__init__(*args, **kwargs)

        params = self.params
        # remove original VPT, install APM
        if hasattr(self, 'vpt'):
            del self.vpt
        self.apm = APM(params, len(self.blocks), self.patch_size, self.embed_dim)

    def forward_features(self, x: torch.Tensor,
                         blur_head_lst=[], target_cls=-1) -> Tuple[torch.Tensor, torch.Tensor]:
        pcam_outputs = None
        self._last_alpha = []          # store mixing weights for analysis

        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        attn_map = None
        for idx, block in enumerate(self.blocks):
            # --- APM: compute dynamic mixed prompt from current CLS token ---
            mixed_prompt, alpha = self.apm.retrieve_prompt(idx, x)
            self._last_alpha.append(alpha.detach())     # (B, K) per layer

            # prepend mixed prompt  (same as Prompt-CAM prepend)
            x = torch.cat([mixed_prompt, x], dim=1)

            # forward block
            if idx == len(self.blocks) - 1:
                x, attn_map = block(x, idx,
                                    blur_head_lst=blur_head_lst,
                                    target_cls=target_cls)
            else:
                x, _ = block(x, idx)

            # strip prompt tokens — same logic as Prompt-CAM
            pcam_outputs = x
            x = x[:, self.params.vpt_num:, :]

        # Prompt-CAM: use full last-layer output (prompts + patches)
        x = pcam_outputs
        x = self.norm(x)
        return x, attn_map

    def forward_head(self, x, pre_logits=False):
        """APM: slice K prompt tokens, pass through head, squeeze to (B, K)."""
        output_feature = x[:, :self.params.vpt_num]   # (B, K, D)
        if pre_logits:
            return output_feature
        out = self.head(output_feature)   # (B, K, 1)
        return out.squeeze(-1)            # (B, K)

    def get_mixing_weights(self):
        """Return per-layer mixing weights from the last forward pass.
        Useful for analysis and visualisation.
        Returns list of (B, K) tensors, one per layer.
        """
        return getattr(self, '_last_alpha', [])


# ---------------------------------------------------------------------------
# Registry helpers — mirror the DINO and DINOv2 entries in vision_transformer
# ---------------------------------------------------------------------------

def _create_vit_apm(variant: str, pretrained: bool = False, **kwargs):
    return build_model_with_cfg(
        VisionTransformerAPM,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_strict=True,
        **kwargs,
    )


@register_model
def vit_base_patch16_dino_apm(pretrained: bool = False, **kwargs):
    """ViT-B/16 DINO with APM."""
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    return _create_vit_apm(
        'vit_base_patch16_224.dino', pretrained=pretrained,
        **dict(model_args, **kwargs))


@register_model
def vit_base_patch14_dinov2_apm(pretrained: bool = False, **kwargs):
    """ViT-B/14 DINOv2 with APM (img_size fixed to 224)."""
    model_args = dict(patch_size=14, embed_dim=768, depth=12,
                      num_heads=12, init_values=1e-5, img_size=224)
    return _create_vit_apm(
        'vit_base_patch14_dinov2', pretrained=pretrained,
        **dict(model_args, **kwargs))
