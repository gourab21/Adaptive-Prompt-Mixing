"""
Model factory for APM variants.
Mirrors experiment/build_model.py but registers APM tune modules.
"""

import timm
import torch
from model.vision_transformer_apm import VisionTransformerAPM  # registers models
from utils.log_utils import log_model_info
from utils.setup_logging import get_logger
import torch.distributed as dist

logger = get_logger("Prompt_CAM")

# APM adds 'apm' module (prompts + mixer) alongside the head
TUNE_MODULES = ['apm', 'head']


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def get_model_apm(params, visualize=False):
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {params.device}")

    model = _get_base_model_apm(params, visualize=visualize)

    tune_parameters = []
    if params.debug and is_main_process():
        logger.info("Trainable params:")

    for name, parameter in model.named_parameters():
        if any(m in name for m in TUNE_MODULES):
            parameter.requires_grad = True
            tune_parameters.append(parameter)
            if params.debug and is_main_process():
                logger.info("\t{}, {}, {}".format(name, parameter.numel(), parameter.shape))
        else:
            parameter.requires_grad = False

    model_grad_params_no_head = log_model_info(model, logger)
    model = model.cuda(device=params.device)
    return model, tune_parameters, model_grad_params_no_head


def _get_base_model_apm(params, visualize=False):
    if params.pretrained_weights == 'vit_base_patch16_dino':
        model = timm.create_model(
            'vit_base_patch16_dino_apm',
            drop_path_rate=params.drop_path_rate,
            pretrained=False,
            params=params)
        if not visualize:
            model.load_pretrained('pretrained_weights/dino_vitbase16_pretrain.pth')
        model.reset_classifier(params.class_num)

    elif params.pretrained_weights == 'vit_base_patch14_dinov2':
        params.patch_size = 14
        model = timm.create_model(
            'vit_base_patch14_dinov2_apm',
            drop_path_rate=params.drop_path_rate,
            pretrained=False,
            params=params)
        if not visualize:
            model.load_pretrained('pretrained_weights/dinov2_vitb14_pretrain.pth')
        model.reset_classifier(params.class_num)

    else:
        raise NotImplementedError(
            f"APM not yet wired for backbone: {params.pretrained_weights}")

    return model
