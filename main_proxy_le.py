import logging
import os

import custom_graphgym  # noqa, register custom modules
import torch

from torch_geometric import seed_everything
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg, load_cfg,
                                             set_agg_dir, set_run_dir)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import create_logger, set_printing
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
from torch_geometric.graphgym.register import train_dict
from torch_geometric.graphgym.proxy_le import proxy_le, attach_randomvec_le
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.golden_model_train import attach_eigen_vec

"""
This pipeline is proxy task using Laplacian Matrix's Eigenvectors as proxy task
"""
if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    load_cfg(cfg, args)  # 这里cfg由命令行进行指定，是一个.yaml文件
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    dump_cfg(cfg)
    # Repeat for different random seeds
    auto_select_device()
    loaders = create_loader()  # list of loaders, they are divided from original dataset according to 'train' 'test' and 'val'
    attach_eigen_vec(loaders)
    for i in range(args.repeat):
        set_run_dir(cfg.out_dir, args.cfg_file)
        set_printing()
        # Set configurations for each run
        cfg.seed = cfg.seed + 1
        # TODO(wby) Here's a problem: is it okay to set different proxy task for each repeat? In another word, can we initialize loader before loop?

        # seed_everything(cfg.seed) # Sets the seed for generating random numbers in PyTorch, numpy and Python.实际上就是生成随机数，避免训练结果相同
        # Set machine learning pipeline
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
        # attach_eigenvec(loaders) # TODO(wby) Here we can choose to attach eigenvector or random vector to loaders as proxy task
        if cfg.train.mode == 'standard':
            proxy_le(loggers, loaders, model, optimizer, scheduler)
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                       scheduler)
    # Aggregate results from different seeds
    agg_runs(set_agg_dir(cfg.out_dir, args.cfg_file), cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
