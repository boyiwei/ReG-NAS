import logging
import time

import torch

from torch_geometric.graphgym.checkpoint import (
    clean_ckpt,
    load_ckpt,
    save_ckpt,
)
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.utils.epoch import (
    is_ckpt_epoch,
    is_eval_epoch,
    is_train_eval_epoch,
)


def train_epoch_inference(logger, loader, model, optimizer, scheduler):
    model.train()
    time_start = time.time()
    for batch in loader:
        batch.split = 'train'
        optimizer.zero_grad()
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        loss.backward()
        optimizer.step()
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(), loss=loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()
    scheduler.step()


@torch.no_grad()
def eval_epoch_inference(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(), loss=loss.item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()


def train_inference(loggers, loaders, model, optimizer, scheduler):
    """
    The core training pipeline
    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler
    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler,
                                cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    cur_epoch = 0
    if is_eval_epoch(cur_epoch):
        split = 'test'
        eval_epoch_inference(loggers[2], loaders[2], model,
                             split=split)
        loggers[2].write_epoch(cur_epoch)
    if is_ckpt_epoch(cur_epoch) and cfg.train.enable_ckpt:
        save_ckpt(model, optimizer, scheduler, cur_epoch)
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.run_dir))
