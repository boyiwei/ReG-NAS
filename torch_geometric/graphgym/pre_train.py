import logging
import time

import torch
import torch_geometric
from torch_geometric.graphgym.checkpoint import (clean_ckpt, load_ckpt, save_ckpt)
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.utils.epoch import (is_ckpt_epoch, is_eval_epoch, is_train_eval_epoch)
from torch_geometric.graphgym.loader import set_dataset_attr_eig
from torch_geometric.graphgym.logger import create_logger

total_feat_out = []
total_feat_in = []

def hook_fn_forward(module, input, output):
    # print("=======================================================================================")
    # print(module) # 用于区分模块
    # print('input', input) # 首先打印出来
    # print('output', output)
    total_feat_out.append(output)
    total_feat_in.append(input)


def compute_laplacian(data): # TODO(wby) try to compute eigenvectors before training!
    adj_matrix = torch_geometric.utils.to_dense_adj(data.edge_index)
    adj_matrix = adj_matrix.squeeze(0)
    adj_matrix = adj_matrix.to(torch.device(cfg.device))
    if adj_matrix.shape[0] != data.num_nodes: # if there are iso-nodes, we need to expand matrix
        # print("error")
        num_iso_nodes = data.num_nodes - adj_matrix.shape[0]
        p1 = torch.zeros(adj_matrix.shape[0], num_iso_nodes, device=torch.device(cfg.device))
        adj_temp = torch.cat((adj_matrix, p1), 1)
        p2 = torch.zeros(num_iso_nodes, adj_temp.shape[1], device=torch.device(cfg.device))
        adj_matrix = torch.cat((adj_temp, p2), 0)
    R = torch.sum(adj_matrix, dim=1)
    degree_matrix = torch.diag(R)
    laplacian_matrix = degree_matrix - adj_matrix
    # 计算特征值以及特征向量
    levals, levecs = torch.linalg.eig(laplacian_matrix)
    levals = levals.real
    levecs = levecs.real
    levecs = levecs.t()
    sorted, indices = torch.sort(levals, descending=True, dim=0)
    cor = indices[0] # find the largest eigenvector and its corresponding coordinate
    max_levec = levecs[cor]
    max_levec = max_levec.unsqueeze(1)
    return max_levec


def proxy_pooling(x):
    y = x.unsqueeze(0)
    # layer = torch.nn.AvgPool2d((x.size(0), 1), stride=1) # transfer[27,300]->[1,300]
    layer = torch.nn.MaxPool2d((1, x.shape[1]), stride=1) # transfer[27,300]->[27,1]
    y = layer(y)
    y = y.squeeze(0)
    return y

def proxy_pooling2(x):
    y = x.unsqueeze(0)
    layer = torch.nn.AvgPool2d((x.size(0), 1), stride=1) # transfer[27,300]->[1,300]
    # layer = torch.nn.MaxPool2d((1, x.shape[1]), stride=1) # transfer[27,300]->[27,1]
    y = layer(y)
    y = y.squeeze(0)
    return y


def compute_loss_proxy(batch): # TODO(wby) : add net_feat into attribute, do not use for!
    """compute the loss, proxy_score and ground truth of proxy task (aka graph Laplacian eigenvector)"""
    # num_graphs = batch.num_graphs
    # compute first dataset in order to define loss function and score
    # data = batch[0]
    net_feat = proxy_pooling(batch.x)
    proxy_vec = batch.eig_vec
    proxy_true = proxy_vec
    loss, proxy_score = compute_loss(net_feat, proxy_vec)
    # for i in range(1, num_graphs):
    #     data = batch[i]
    #     net_feat = proxy_pooling(data.x)
    #     proxy_vec = compute_laplacian(data)
    #     loss_data, proxy_score_data = compute_loss(net_feat, proxy_vec)
    #     loss = loss + loss_data
    #     proxy_score = torch.cat([proxy_score,proxy_score_data], dim=0)
    #     proxy_true = torch.cat([proxy_true,proxy_vec], dim=0)
    return loss, proxy_score,proxy_true


def compute_loss_proxy2(batch): # TODO(wby) : add net_feat into attribute, do not use for!
    """compute the loss, proxy_score and ground truth of proxy task (aka graph Laplacian eigenvector)"""
    # num_graphs = batch.num_graphs
    # compute first dataset in order to define loss function and score
    # data = batch[0]
    net_feat = proxy_pooling2(batch.x)
    proxy_vec = batch.eig_vec
    proxy_true = proxy_vec
    loss, proxy_score = compute_loss(net_feat, proxy_vec)
    # for i in range(1, num_graphs):
    #     data = batch[i]
    #     net_feat = proxy_pooling(data.x)
    #     proxy_vec = compute_laplacian(data)
    #     loss_data, proxy_score_data = compute_loss(net_feat, proxy_vec)
    #     loss = loss + loss_data
    #     proxy_score = torch.cat([proxy_score,proxy_score_data], dim=0)
    #     proxy_true = torch.cat([proxy_true,proxy_vec], dim=0)
    return loss, proxy_score,proxy_true


def attach_randomvec(loaders):
    indices0 = loaders[0].dataset._indices
    indices1 = loaders[1].dataset._indices
    indices2 = loaders[2].dataset._indices
    full_length = indices0.__len__() + indices1.__len__() + indices2.__len__()
    for i in range (0, full_length):
        if i in indices0:
            loader = loaders[0]
            indices = indices0
        elif i in indices1:
            loader = loaders[1]
            indices = indices1
        else:
            loader = loaders[2]
            indices = indices2
        dataset = loader.dataset
        j = indices.index(i)
        data = dataset[j]
        size = data.x.shape[0]
        rand_vec = 2 * torch.rand([size, 1]) - 1  # In order to make the random vector distribute in range [-1, 1]
        rand_vec = rand_vec / torch.norm(rand_vec)
        if i==0:
            # evec_all = compute_laplacian(data)
            rand_all = 2 * torch.rand([size, 1]) - 1
            rand_all = rand_all / torch.norm(rand_all)
        else:
            rand_all = torch.cat([rand_all, rand_vec], dim=0)
    rand_all = rand_all.to(torch.device('cpu'))
    slice = dataset.slices['x']
    for i in range(0, loaders.__len__()):
        loader = loaders[i]
        set_dataset_attr_eig(loader.dataset,'eig_vec', rand_all, slice)


def attach_randomvec2(loaders):
    indices0 = loaders[0].dataset._indices
    indices1 = loaders[1].dataset._indices
    indices2 = loaders[2].dataset._indices
    full_length = indices0.__len__() + indices1.__len__() + indices2.__len__()
    for i in range (0, full_length):
        if i in indices0:
            loader = loaders[0]
            indices = indices0
        elif i in indices1:
            loader = loaders[1]
            indices = indices1
        else:
            loader = loaders[2]
            indices = indices2
        dataset = loader.dataset
        j = indices.index(i)
        data = dataset[j]
        size = cfg.gnn.dim_inner
        rand_vec = 2 * torch.rand([size, 1]) - 1  # In order to make the random vector distribute in range [-1, 1]
        rand_vec = rand_vec / torch.norm(rand_vec)
        if i==0:
            # evec_all = compute_laplacian(data)
            rand_all = 2 * torch.rand([size, 1]) - 1
            rand_all = rand_all / torch.norm(rand_all)
        else:
            rand_all = torch.cat([rand_all, rand_vec], dim=0)
    rand_all = rand_all.to(torch.device('cpu'))
    slice = dataset.slices['x']
    for i in range(0, loaders.__len__()):
        loader = loaders[i]
        set_dataset_attr_eig(loader.dataset,'eig_vec', rand_all, slice)



def attach_eigenvec(loaders):
    indices0 = loaders[0].dataset._indices
    indices1 = loaders[1].dataset._indices
    indices2 = loaders[2].dataset._indices
    full_length = indices0.__len__() + indices1.__len__() + indices2.__len__()
    for i in range(0, full_length):
        if i in indices0:
            loader = loaders[0]
            indices = indices0
        elif i in indices1:
            loader = loaders[1]
            indices = indices1
        else:
            loader = loaders[2]
            indices = indices2
        dataset = loader.dataset
        j = indices.index(i)
        data = dataset[j]
        evec_single = compute_laplacian(data)
        if i == 0:
            evec_all = compute_laplacian(data)
        else:
            evec_all = torch.cat([evec_all, evec_single], dim=0)
    evec_all = evec_all.to(torch.device('cpu'))
    slice = dataset.slices['x']
    for i in range(0, loaders.__len__()):
        loader = loaders[i]
        set_dataset_attr_eig(loader.dataset, 'eig_vec', evec_all, slice)


def train_epoch(logger, loader, model, optimizer, scheduler):
    model.train()
    time_start = time.time()
    for batch in loader:
        batch.split = 'train'
        optimizer.zero_grad()
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        del total_feat_in[-1]
        del total_feat_out[-1]  # delete item in order to save gpu memory
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


def proxy_epoch(logger, loader, model, optimizer, scheduler):
    global total_feat_in
    global total_feat_out
    model.train()
    time_start = time.time()
    for batch in loader:
        batch.split = 'train'
        optimizer.zero_grad()
        batch.to(torch.device(cfg.device))
        pred_train, true_train = model(batch)
        batch_hid = total_feat_in[-1][0] # extract hidden layer feature
        del total_feat_in[-1]
        del total_feat_out[-1]  # delete item in order to save gpu memory
        loss, proxy_score, proxy_true = compute_loss_proxy(batch_hid) # TODO (wby) here we redefined the compute_loss function
        # loss, pred_score = compute_loss(pred_train, true_train)
        loss.backward()
        optimizer.step()
        logger.update_stats(true=proxy_true.detach().cpu(),
                            pred=proxy_score.detach().cpu(), loss=loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()
    scheduler.step()


def proxy_epoch2(logger, loader, model, optimizer, scheduler):
    global total_feat_in
    global total_feat_out
    model.train()
    time_start = time.time()
    for batch in loader:
        batch.split = 'train'
        optimizer.zero_grad()
        batch.to(torch.device(cfg.device))
        pred_train, true_train = model(batch)
        batch_hid = total_feat_in[-1][0] # extract hidden layer feature
        del total_feat_in[-1]
        del total_feat_out[-1]  # delete item in order to save gpu memory
        loss, proxy_score, proxy_true = compute_loss_proxy(batch_hid) # TODO (wby) here we redefined the compute_loss function
        # loss, pred_score = compute_loss(pred_train, true_train)
        loss.backward()
        optimizer.step()
        logger.update_stats(true=proxy_true.detach().cpu(),
                            pred=proxy_score.detach().cpu(), loss=loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()
    scheduler.step()


@torch.no_grad()
def proxy_eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()
    """Register hook function is not needed here, because we have registered it on train_epoch function, and module will be
    hooked twice if we registered again."""
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.device))
        pred_eval, true_eval = model(batch)
        batch_hid = total_feat_in[-1][0]  # extract hidden layer feature
        del total_feat_in[-1]
        del total_feat_out[-1]  # delete item in order to save gpu memory
        loss, proxy_score, proxy_true = compute_loss_proxy(batch_hid)
        # loss, pred_score = compute_loss(pred, true)
        logger.update_stats(true=proxy_true.detach().cpu(),
                            pred=proxy_score.detach().cpu(), loss=loss.item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        del total_feat_in[-1]
        del total_feat_out[-1]  # delete item in order to save gpu memory
        loss, pred_score = compute_loss(pred, true)
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(), loss=loss.item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()


def pre_train(loggers, loaders, model, optimizer, scheduler):
    """
    The core proxy training pipeline

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    for name0, module0 in model.named_children():
        if name0=='post_mp':
            module0.register_forward_hook(hook_fn_forward)      # hook function to extract features from hidden layer

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
    '''
    pre-train process. In this part we want to use random proxy task to pre-train 20 epoch, then 
    use groundtruth task to train 80 epoch in order to get better loss and higher auc
    '''
    for cur_epoch in range(start_epoch, 20):
        proxy_epoch(loggers[0], loaders[0], model, optimizer, scheduler)
        '''Determines if the model should be evaluated at the training epoch. The difference between is_train_eval_epoch
        and is_eval_epoch is that the user can self define whether logger should or not record the train-process-data
        (aka cfg.train.skip_train_eval), if cfg.train.skip_train_eval is True, loggers would record training-data
        '''
        if is_train_eval_epoch(cur_epoch):
            loggers[0].write_epoch(cur_epoch)   # logger write for train datasets
        if is_eval_epoch(cur_epoch):            # Determines if the model should be evaluated at the current epoch.
            for i in range(1, num_splits):
                proxy_eval_epoch(loggers[i], loaders[i], model,
                           split=split_names[i - 1])
                loggers[i].write_epoch(cur_epoch)
        if is_ckpt_epoch(cur_epoch) and cfg.train.enable_ckpt:
            save_ckpt(model, optimizer, scheduler, cur_epoch)

    """
    post-train process. In this part we would train 80 epoch and get a better loss
    """
    # redefine loggers and loss functions
    cfg.model.loss_fun = 'cross_entropy'
    cfg.dataset.task_type = 'classification'
    loggers = create_logger()
    # TODO(wby) Here we need to refine the code!

    for cur_epoch in range(20, cfg.optim.max_epoch):
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler)
        if is_train_eval_epoch(cur_epoch):
            loggers[0].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model,
                           split=split_names[i - 1])
                loggers[i].write_epoch(cur_epoch)
        if is_ckpt_epoch(cur_epoch) and cfg.train.enable_ckpt:
            save_ckpt(model, optimizer, scheduler, cur_epoch)

    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.run_dir))
