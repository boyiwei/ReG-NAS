import logging
import os

import custom_graphgym  # noqa, register custom modules
import torch

from torch_geometric import seed_everything
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg, load_cfg,
                                             set_agg_dir, set_run_dir)
from torch_geometric.graphgym.loader import get_loader
from torch_geometric.graphgym.logger import create_logger, set_printing
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
from torch_geometric.graphgym.register import train_dict
from torch_geometric.graphgym.train import train
from torch_geometric.graphgym.utils.agg_runs import agg_runs, agg_runs_inference
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.datasets import ZINC


def create_loader_ZINC():
    """
    Create data loader object

    Returns: List of PyTorch data loaders

    """
    dataset_train = ZINC(root='./datasets/ZINC', split='train', subset=True)
    dataset_test = ZINC(root='./datasets/ZINC/', split='test', subset=True)
    dataset_val = ZINC(root='./datasets/ZINC/', split='val', subset=True)
    # train loader

    loaders = [get_loader(dataset_train, cfg.train.sampler, cfg.train.batch_size,
                       shuffle=True)]

    # val and test loaders
    loaders.append(
        get_loader(dataset_val, cfg.val.sampler, cfg.train.batch_size,
                   shuffle=False))

    loaders.append(
        get_loader(dataset_test, cfg.val.sampler, cfg.train.batch_size,
                   shuffle=False))

    return loaders

if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    load_cfg(cfg, args)  # 这里cfg由命令行进行指定，是一个.yaml文件
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    dump_cfg(cfg)
    # Repeat for different random seeds
    for i in range(args.repeat):
        set_run_dir(cfg.out_dir, args.cfg_file)
        set_printing()
        # Set configurations for each run
        cfg.seed = cfg.seed + 1
        cfg.share.num_splits = 3
        # seed_everything(cfg.seed) # Sets the seed for generating random numbers in PyTorch, numpy and Python.实际上就是生成随机数，避免训练结果相同
        auto_select_device()
        # Set machine learning pipeline
        loaders = create_loader_ZINC()
        loggers = create_logger()
        model = create_model()
        optimizer = create_optimizer(model.parameters(), cfg.optim)
        scheduler = create_scheduler(optimizer, cfg.optim)
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        # Start training
        if cfg.train.mode == 'standard':
            train(loggers, loaders, model, optimizer, scheduler) # TODO(wby): The core training pipeline
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                       scheduler)
    # Aggregate results from different seeds
    agg_runs(set_agg_dir(cfg.out_dir, args.cfg_file), cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
