import argparse
from experiment.run_apm import basic_run_apm
from utils.setup_logging import get_logger
from utils.misc import set_seed, load_yaml, override_args_with_yaml
import time

logger = get_logger("Prompt_CAM")


def main():
    args = setup_parser().parse_args()

    if args.config:
        yaml_config = load_yaml(args.config)
        if yaml_config:
            args = override_args_with_yaml(args, yaml_config)

    set_seed(args.random_seed)
    start = time.time()
    args.vis_attn = False
    basic_run_apm(args)
    end = time.time()
    logger.info(f'----------- Total Run time : {(end - start) / 60} mins-----------')


def setup_parser():
    parser = argparse.ArgumentParser(description='APM — Adaptive Prompt Mixing')

    parser.add_argument('--pretrained_weights', type=str,
                        default='vit_base_patch16_dino',
                        choices=['vit_base_patch16_dino', 'vit_base_patch14_dinov2'],
                        help='Backbone pretrained weights')
    parser.add_argument('--drop_path_rate', default=0.1, type=float)
    parser.add_argument('--model', type=str, default='dino',
                        choices=['dino', 'dinov2'])
    parser.add_argument('--train_type', type=str, default='apm',
                        choices=['apm'],
                        help='Must be apm for this entry point')

    # Optimizer / scheduler
    parser.add_argument('--optimizer', default='sgd',
                        choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--warmup_epoch', default=20, type=int)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--warmup_lr_init', type=float, default=0)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--test_batch_size', default=32, type=int)
    parser.add_argument('--wd', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--early_patience', type=int, default=101)

    # Data
    parser.add_argument('--data', default='cub')
    parser.add_argument('--data_path', default='./data/images/cub')
    parser.add_argument('--crop_size', default=224, type=int)
    parser.add_argument('--final_run', action='store_false')
    parser.add_argument('--normalized', action='store_false')

    # Prompt (vpt_num = num classes, same as Prompt-CAM)
    parser.add_argument('--vpt_mode', type=str, default=None)
    parser.add_argument('--vpt_num', default=200, type=int,
                        help='Number of class prompts (= num_classes)')
    parser.add_argument('--vpt_layer', default=None, type=int)
    parser.add_argument('--vpt_dropout', default=0.1, type=float)

    # APM v2/v3: deletion-sharpening regularizers (see engine/trainer.py)
    parser.add_argument('--entropy_weight', default=0.05, type=float,
                        help='Weight of the saliency-entropy penalty on the '
                             'ground-truth class attention row over patches. '
                             '0 disables it (reduces to v1 APM behavior).')
    parser.add_argument('--entropy_warmup_epochs', default=20, type=int,
                        help='[v3] Linearly ramp entropy_weight in from 0 over '
                             'this many epochs instead of applying it at full '
                             'strength from epoch 0. 0 = no ramp (v2 behavior).')
    parser.add_argument('--topk_weight', default=0.02, type=float,
                        help='[v3] Weight of the top-k mass penalty: rewards '
                             'concentrating saliency mass into the row\'s own '
                             'top-k patches. 0 disables it.')
    parser.add_argument('--topk_frac', default=0.1, type=float,
                        help='[v3] Fraction of the N patch tokens counted as '
                             '"top-k" for the top-k mass penalty.')

    # APM v4: margin loss directly targeting deletion's actual mechanic
    parser.add_argument('--margin_weight', default=0.05, type=float,
                        help='[v4] Weight of the top-1-vs-rest margin/hinge '
                             'loss on the saliency row. 0 disables it.')
    parser.add_argument('--margin', default=0.05, type=float,
                        help='[v4] Required gap between the row\'s top-1 '
                             'value and its margin_frac-th percentile value.')
    parser.add_argument('--margin_frac', default=0.5, type=float,
                        help='[v4] Percentile (as a fraction of N patches) '
                             'the top-1 value is compared against.')

    # Misc
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--gpu_num', default=1, type=int)
    parser.add_argument('--debug', action='store_false')
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--eval_freq', default=10, type=int)
    parser.add_argument('--store_ckp', action='store_true')
    parser.add_argument('--final_acc_hp', action='store_false')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')

    return parser


if __name__ == '__main__':
    main()
