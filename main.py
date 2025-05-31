import multiprocessing as mp
import argparse
import os
from pathlib import Path
import yaml
import utils
from utils import dist_init, distributed_utils
from trainer import Trainer
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in config.items():
        setattr(args, k, v)  

    dist_init(args)

    # mkdir path
    output_dir = Path(args.exp_path).resolve()
    output_dir = utils.common_utils.increment_path(output_dir, exist_ok=False)
    output_dir.mkdir(parents=True, exist_ok=True)  # make dir
    print(f"output path:{output_dir}")
    args.exp_path = str(output_dir)

    (output_dir/"checkpoints").mkdir(exist_ok=True)
    (output_dir/"images").mkdir(exist_ok=True)
    (output_dir/"events").mkdir(exist_ok=True)
    flogger = utils.create_file_logger("global_logger", f"{args.exp_path}/log.txt")

    if distributed_utils.is_main_process():
        print('--------user config:')
        flogger.info('--------user config:')
        for k, v in args.__dict__.items():
            if not k.startswith('_'):
                _ss = "%-30s: %-20s" % (k, getattr(args, k))
                print(_ss)
                flogger.info(_ss)
        print('--------------------')
        flogger.info('--------------------')
    args.flogger = flogger
    # train
    trainer = Trainer(args)
    trainer.run()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch De-Occlusion.')
    parser.add_argument('--config',  type=str, default=ROOT/'configs/config_kwob.yaml')
    parser.add_argument('--evaluate', default=False, type=bool,  help='only evaluate')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--load_pretrain', default=ROOT/'output/result/checkpoints/ckp20.pth', type=str)
    parser.add_argument('--load_epoch', default=20, type=int, help='load_epoch >=1')
    parser.add_argument('--exp_path', type=str, default='./output/result')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()

    main(args)
