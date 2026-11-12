"""
APM run entry — identical to experiment/run.py except:
  - imports get_model_apm instead of get_model
  - train_type 'apm' is recognised by build_model_apm
"""
import os
import torch
import yaml
import torch.distributed as dist
from datetime import datetime
from timm.utils import get_outdir

from experiment.build_model_apm import get_model_apm
from experiment.build_loader import get_loader
from utils.global_var import OUTPUT_DIR
from engine.trainer import Trainer
from utils.log_utils import logging_env_setup
from utils.misc import method_name
from utils.setup_logging import get_logger

logger = get_logger("Prompt_CAM")


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return True
    return False


def train_apm(params, train_loader, val_loader, test_loader):
    model, tune_parameters, model_grad_params_no_head = get_model_apm(params)

    device = torch.device("cuda", params.local_rank)
    model = model.to(device)

    if params.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[params.local_rank],
            output_device=params.local_rank,
            find_unused_parameters=True,
        )

    trainer = Trainer(model, tune_parameters, params)
    train_metrics, best_eval_metrics, eval_metrics = trainer.train_classifier(
        train_loader, val_loader, test_loader)
    return train_metrics, best_eval_metrics, eval_metrics, model_grad_params_no_head


def basic_run_apm(params):
    is_distributed = init_distributed()
    params.distributed = is_distributed
    params.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    data_name = params.data.split("-")[-1]
    dataset_name = params.data.split("-")[0]
    method = method_name(params)
    start_time = datetime.now().strftime("%Y-%m-%d-%H:%M")
    output_dir = os.path.join(OUTPUT_DIR, params.pretrained_weights,
                              dataset_name, method, data_name, start_time)
    params.output_dir = get_outdir(output_dir)

    params_text = yaml.safe_dump(params.__dict__, default_flow_style=False)
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(params_text)

    logging_env_setup(params)
    params.class_num = params.vpt_num
    logger.info(f'Start loading {data_name}')
    train_loader, val_loader, test_loader = get_loader(params, logger)
    train_apm(params, train_loader, val_loader, test_loader)
