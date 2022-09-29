import logging
import os
import argparse
import custom_graphgym  # noqa, register custom modules
import torch
import torch.nn.functional as F
from copy import deepcopy
from torch_geometric import seed_everything
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg, load_cfg,
                                             set_agg_dir, set_run_dir)
from torch_geometric.graphgym.loader import set_dataset_attr_eig
from torch_geometric.nn.glob.glob import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import create_logger, set_printing
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
from torch_geometric.graphgym.register import train_dict
from torch_geometric.graphgym.train import train
from torch_geometric.nn.glob.glob import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.loader import get_loader, create_dataset
from main_ZINC import create_loader_ZINC
from torch_geometric.datasets import ZINC
from torch_geometric.graphgym.proxy import compute_laplacian

total_feat_out = []


def hook_fn_forward(module, input, output):
    if type(output) is tuple:
        output_clone = output[-1].clone()
    else:
        output_clone = output.clone()
    total_feat_out.append(output_clone)


def create_sequential_loader():
    """
    Create data loader object

    Returns: List of PyTorch data loaders

    """
    dataset = create_dataset()
    # train loader
    if cfg.dataset.task == 'graph':
        id = dataset.data['train_graph_index']
        loaders = [
            get_loader(dataset[id], cfg.train.sampler, cfg.train.batch_size,
                       shuffle=False)
        ]
        delattr(dataset.data, 'train_graph_index')
    else:
        loaders = [
            get_loader(dataset, cfg.train.sampler, cfg.train.batch_size,
                       shuffle=False)
        ]

    # val and test loaders
    for i in range(cfg.share.num_splits - 1):
        if cfg.dataset.task == 'graph':
            split_names = ['val_graph_index', 'test_graph_index']
            id = dataset.data[split_names[i]]
            loaders.append(
                get_loader(dataset[id], cfg.val.sampler, cfg.train.batch_size,
                           shuffle=False))
            delattr(dataset.data, split_names[i])
        else:
            loaders.append(
                get_loader(dataset, cfg.val.sampler, cfg.train.batch_size,
                           shuffle=False))

    return loaders


def create_sequential_loader_ZINC():
    """
    Create data loader object

    Returns: List of PyTorch data loaders

    """
    dataset_train = ZINC(root='./datasets/ZINC/', split='train', subset=True)
    dataset_test = ZINC(root='./datasets/ZINC/', split='test', subset=True)
    dataset_val = ZINC(root='./datasets/ZINC/', split='val', subset=True)
    # train loader

    loaders = [get_loader(dataset_train, cfg.train.sampler, cfg.train.batch_size,
                       shuffle=False)]

    # val and test loaders
    loaders.append(
        get_loader(dataset_val, cfg.val.sampler, cfg.train.batch_size,
                   shuffle=False))

    loaders.append(
        get_loader(dataset_test, cfg.val.sampler, cfg.train.batch_size,
                   shuffle=False))

    return loaders


def golden_model_train(args):
    # Load config file
    load_cfg(cfg, args)  # 这里cfg由命令行进行指定，是一个.yaml文件
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    dump_cfg(cfg)
    # Repeat for different random seeds
    set_run_dir(cfg.out_dir, args.cfg_file)
    set_printing()
    # Set configurations for each run
    cfg.seed = cfg.seed + 1
    auto_select_device()
    # Set machine learning pipeline
    loaders = create_loader()
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
        train(loggers, loaders, model, optimizer, scheduler)
    return model


def golden_model_train_ZINC(args):
    # Load config file
    load_cfg(cfg, args)  # 这里cfg由命令行进行指定，是一个.yaml文件
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    dump_cfg(cfg)
    # Repeat for different random seeds
    set_run_dir(cfg.out_dir, args.cfg_file)
    set_printing()
    # Set configurations for each run
    cfg.seed = cfg.seed + 1
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
        train(loggers, loaders, model, optimizer, scheduler)
    return model


def create_golden_vec(pooling='add', repeat=1, method='golden'):
    """
    和create_golden_vec类似，都是创建proxy task，区别在于本函数采用了hook函数的方法来提取中间层结果
    :return:
    """
    cfg.model.graph_pooling = pooling
    if method == 'golden':
        args = argparse.Namespace(cfg_file='configs/personal/golden_model.yaml', mark_done=False, opts=[], repeat=1)
        for n in range(0, repeat):
            model = golden_model_train(args)
            # extract given model layers.
            # optimizer.zero_grad()
            for name0, module0 in model.named_children():
                if name0 == 'post_mp':
                    for name1, module1 in module0.named_children():
                        for name2, module2 in module1.named_children():
                            for name3, module3 in module2.named_children():
                                if name3 == '0':
                                    module3.register_forward_hook(hook_fn_forward)
                # elif name0 == 'mp':
                #     for name1, module1 in module0.named_children():
                #         module1.register_forward_hook(hook_fn_forward)
            model.eval()
            # TODO(wby) Here we need to use this model and conduct reference in order to generate proxy vector
            loaders = create_sequential_loader()
            indices0 = loaders[0].dataset._indices
            indices1 = loaders[1].dataset._indices
            indices2 = loaders[2].dataset._indices
            full_length = indices0.__len__() + indices1.__len__() + indices2.__len__()
            proxy_all = torch.zeros(full_length, 64, device=torch.device(cfg.device))
            for i in range(0, 3):
                loader = loaders[i]
                j = 0
                for batch in loader:
                    batch.to(torch.device(cfg.device))
                    model(batch)
                    batch_hid = total_feat_out[0].clone()
                    del total_feat_out[0]
                    if j == 0:
                        proxy_loader = batch_hid.clone()  # 注意，这里默认的操作是global_add_pool，所以出来之后的结果和global_mean_pool的结果会有差别！
                    else:
                        proxy_vec = batch_hid.clone()
                        proxy_loader = torch.cat([proxy_loader, proxy_vec], dim=0)
                    j = j + 1
                proxy_all[loaders[i].dataset._indices] = proxy_loader
            name0 = pooling
            name1 = str(n)
            name = 'proxy_all_' + name0 + '_golden_' + name1 + '.pt'
            path = os.path.join('datasets/ogbg_molhiv/', name)
            # torch.save(proxy_all, "./datasets/ogbg_molhiv/proxy_all.pt")
            torch.save(proxy_all, path)
    elif method == 'bad':
        args = argparse.Namespace(
            cfg_file='configs/personal/bad_model.yaml',
            mark_done=False, opts=[], repeat=1)
        for n in range(0, repeat):
            model = golden_model_train(args)
            # extract given model layers.
            # optimizer.zero_grad()
            for name0, module0 in model.named_children():
                if name0 == 'post_mp':
                    for name1, module1 in module0.named_children():
                        for name2, module2 in module1.named_children():
                            for name3, module3 in module2.named_children():
                                if name3 == '0':
                                    module3.register_forward_hook(hook_fn_forward)
                # elif name0 == 'mp':
                #     for name1, module1 in module0.named_children():
                #         module1.register_forward_hook(hook_fn_forward)
            model.eval()
            # TODO(wby) Here we need to use this model and conduct reference in order to generate proxy vector
            loaders = create_sequential_loader()
            indices0 = loaders[0].dataset._indices
            indices1 = loaders[1].dataset._indices
            indices2 = loaders[2].dataset._indices
            full_length = indices0.__len__() + indices1.__len__() + indices2.__len__()
            proxy_all = torch.zeros(full_length, 64, device=torch.device(cfg.device))
            for i in range(0, 3):
                loader = loaders[i]
                j = 0
                for batch in loader:
                    batch.to(torch.device(cfg.device))
                    model(batch)
                    batch_hid = total_feat_out[0].clone()
                    del total_feat_out[0]
                    if j == 0:
                        proxy_loader = batch_hid.clone()  # 注意，这里默认的操作是global_add_pool，所以出来之后的结果和global_mean_pool的结果会有差别！
                    else:
                        proxy_vec = batch_hid.clone()
                        proxy_loader = torch.cat([proxy_loader, proxy_vec], dim=0)
                    j = j + 1
                proxy_all[loaders[i].dataset._indices] = proxy_loader
            name0 = pooling
            name1 = str(n)
            name = 'proxy_all_' + name0 + '_bad_' + name1 + '.pt'
            path = os.path.join('datasets/ogbg_molhiv/', name)
            # torch.save(proxy_all, "./datasets/ogbg_molhiv/proxy_all.pt")
            torch.save(proxy_all, path)
    elif method == 'common':
        args = argparse.Namespace(
            cfg_file='configs/personal/common_model.yaml',
            mark_done=False, opts=[], repeat=1)
        for n in range(0, repeat):
            model = golden_model_train(args)
            # extract given model layers.
            # optimizer.zero_grad()
            for name0, module0 in model.named_children():
                if name0 == 'post_mp':
                    for name1, module1 in module0.named_children():
                        for name2, module2 in module1.named_children():
                            for name3, module3 in module2.named_children():
                                if name3 == '0':
                                    module3.register_forward_hook(hook_fn_forward)
                # elif name0 == 'mp':
                #     for name1, module1 in module0.named_children():
                #         module1.register_forward_hook(hook_fn_forward)
            model.eval()
            # TODO(wby) Here we need to use this model and conduct reference in order to generate proxy vector
            loaders = create_sequential_loader()
            indices0 = loaders[0].dataset._indices
            indices1 = loaders[1].dataset._indices
            indices2 = loaders[2].dataset._indices
            full_length = indices0.__len__() + indices1.__len__() + indices2.__len__()
            proxy_all = torch.zeros(full_length, 64, device=torch.device(cfg.device))
            for i in range(0, 3):
                loader = loaders[i]
                j = 0
                for batch in loader:
                    batch.to(torch.device(cfg.device))
                    model(batch)
                    batch_hid = total_feat_out[0].clone()
                    del total_feat_out[0]
                    if j == 0:
                        proxy_loader = batch_hid.clone()  # 注意，这里默认的操作是global_add_pool，所以出来之后的结果和global_mean_pool的结果会有差别！
                    else:
                        proxy_vec = batch_hid.clone()
                        proxy_loader = torch.cat([proxy_loader, proxy_vec], dim=0)
                    j = j + 1
                proxy_all[loaders[i].dataset._indices] = proxy_loader
            name0 = pooling
            name1 = str(n)
            name = 'proxy_all_' + name0 + '_common_' + name1 + '.pt'
            path = os.path.join('datasets/ogbg_molhiv/', name)
            # torch.save(proxy_all, "./datasets/ogbg_molhiv/proxy_all.pt")
            torch.save(proxy_all, path)


def create_eigen_vec(loaders):
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

    name = 'proxy_all_eigen'+ '.pt'
    path = os.path.join('datasets/ogbg_molhiv/', name)
    torch.save(evec_all, path)


def create_golden_vec2(pooling='add', repeat=1):
    args = argparse.Namespace(cfg_file='configs/personal/golden_model.yaml', mark_done=False, opts=[], repeat=1)
    for n in range(0, repeat):
        model = golden_model_train(args)
        # extract given model layers.
        # optimizer.zero_grad()
        model.eval()
        model_encoder = model.encoder
        model_mp = model.mp
        model_postmp_hid = model.post_mp.layer_post_mp.model[0]
        # TODO(wby) Here we need to use this model and conduct reference in order to generate proxy vector
        loaders = create_sequential_loader()
        indices0 = loaders[0].dataset._indices
        indices1 = loaders[1].dataset._indices
        indices2 = loaders[2].dataset._indices
        full_length = indices0.__len__() + indices1.__len__() + indices2.__len__()
        proxy_all = torch.zeros(full_length, 64, device=torch.device(cfg.device))
        for i in range(0, 3):
            loader = loaders[i]
            j = 0
            for batch in loader:
                batch.to(torch.device(cfg.device))
                model_encoder(batch)
                model_mp(batch)
                model_postmp_hid(batch)
                if j == 0:
                    if pooling == 'mean':
                        proxy_loader = global_mean_pool(batch.x, batch.batch)
                    elif pooling == 'add':
                        proxy_loader = global_add_pool(batch.x, batch.batch)
                    elif pooling == 'max':
                        proxy_loader = global_max_pool(batch.x, batch.batch)
                else:
                    if pooling == 'mean':
                        proxy_vec = global_mean_pool(batch.x, batch.batch)
                    elif pooling == 'add':
                        proxy_vec = global_add_pool(batch.x, batch.batch)
                    elif pooling == 'max':
                        proxy_vec = global_max_pool(batch.x, batch.batch)
                    # proxy_vec = global_mean_pool(batch.x, batch.batch)
                    proxy_loader = torch.cat([proxy_loader, proxy_vec], dim=0)
                j = j + 1
            proxy_all[loaders[i].dataset._indices] = proxy_loader
        name0 = pooling
        name1 = str(n)
        name = 'proxy_all_' + name0 + '_golden_' + name1 + '.pt'
        path = os.path.join('datasets/ogbg_molhiv/', name)
        # torch.save(proxy_all, "./datasets/ogbg_molhiv/proxy_all.pt")
        torch.save(proxy_all, path)

    # length = len(dataset._data_list) # 41127
    # slice = torch.linspace(0, length * 64, steps=length+1)
    # for i in range(0, loaders.__len__()):
    #     loader = loaders[i]
    #     set_dataset_attr_eig(loader.dataset, 'eig_vec', proxy_all, slice)


def create_golden_vec_ZINC(pooling='add', repeat=1, method='golden'):
    """
    和create_golden_vec类似，都是创建proxy task，区别在于本函数采用了hook函数的方法来提取中间层结果
    :return:
    """
    cfg.model.graph_pooling = pooling
    if method == 'golden':
        args = argparse.Namespace(cfg_file='configs/personal/golden_model_ZINC.yaml', mark_done=False, opts=[], repeat=1)
        for n in range(0, repeat):
            model = golden_model_train_ZINC(args)
            # extract given model layers.
            # optimizer.zero_grad()
            for name0, module0 in model.named_children():
                if name0 == 'post_mp':
                    for name1, module1 in module0.named_children():
                        for name2, module2 in module1.named_children():
                            for name3, module3 in module2.named_children():
                                if name3 == '0':
                                    module3.register_forward_hook(hook_fn_forward)
                # elif name0 == 'mp':
                #     for name1, module1 in module0.named_children():
                #         module1.register_forward_hook(hook_fn_forward)
            model.eval()
            # TODO(wby) Here we need to use this model and conduct reference in order to generate proxy vector
            loaders = create_sequential_loader_ZINC()
            indices = []
            indices.append(loaders[0].dataset.data.y.shape[0])
            indices.append(loaders[1].dataset.data.y.shape[0])
            indices.append(loaders[2].dataset.data.y.shape[0])
            proxy_all = []
            # proxy_all.append(torch.zeros(indices0, 64, device=torch.device(cfg.device))
            for i in range(0, 3):
                loader = loaders[i]
                j = 0
                for batch in loader:
                    batch.to(torch.device(cfg.device))
                    model(batch)
                    batch_hid = total_feat_out[0].clone()
                    del total_feat_out[0]
                    if j == 0:
                        proxy_loader = batch_hid.clone()  # 注意，这里默认的操作是global_add_pool，所以出来之后的结果和global_mean_pool的结果会有差别！
                    else:
                        proxy_vec = batch_hid.clone()
                        proxy_loader = torch.cat([proxy_loader, proxy_vec], dim=0)
                    j = j + 1
                # proxy_all[loaders[i].dataset._indices] =
                proxy_all.append(proxy_loader)
            name0 = pooling
            name1 = str(n)
            name = 'proxy_all_' + name0 + '_golden_ZINC_' + name1 + '.pt'
            path = os.path.join('datasets/ZINC/', name)
            # torch.save(proxy_all, "./datasets/ogbg_molhiv/proxy_all.pt")
            torch.save(proxy_all, path)
    elif method == 'bad':
        args = argparse.Namespace(
            cfg_file='configs/personal/bad_model_ZINC.yaml',
            mark_done=False, opts=[], repeat=1)
        for n in range(0, repeat):
            model = golden_model_train_ZINC(args)
            # extract given model layers.
            # optimizer.zero_grad()
            for name0, module0 in model.named_children():
                if name0 == 'post_mp':
                    for name1, module1 in module0.named_children():
                        for name2, module2 in module1.named_children():
                            for name3, module3 in module2.named_children():
                                if name3 == '0':
                                    module3.register_forward_hook(hook_fn_forward)
                # elif name0 == 'mp':
                #     for name1, module1 in module0.named_children():
                #         module1.register_forward_hook(hook_fn_forward)
            model.eval()
            # TODO(wby) Here we need to use this model and conduct reference in order to generate proxy vector
            loaders = create_sequential_loader_ZINC()
            indices0 = loaders[0].dataset._indices
            indices1 = loaders[1].dataset._indices
            indices2 = loaders[2].dataset._indices
            full_length = indices0.__len__() + indices1.__len__() + indices2.__len__()
            proxy_all = torch.zeros(full_length, 64, device=torch.device(cfg.device))
            for i in range(0, 3):
                loader = loaders[i]
                j = 0
                for batch in loader:
                    batch.to(torch.device(cfg.device))
                    model(batch)
                    batch_hid = total_feat_out[0].clone()
                    del total_feat_out[0]
                    if j == 0:
                        proxy_loader = batch_hid.clone()  # 注意，这里默认的操作是global_add_pool，所以出来之后的结果和global_mean_pool的结果会有差别！
                    else:
                        proxy_vec = batch_hid.clone()
                        proxy_loader = torch.cat([proxy_loader, proxy_vec], dim=0)
                    j = j + 1
                proxy_all[loaders[i].dataset._indices] = proxy_loader
            name0 = pooling
            name1 = str(n)
            name = 'proxy_all_' + name0 + '_bad_ZINC_' + name1 + '.pt'
            path = os.path.join('datasets/ZINC/', name)
            # torch.save(proxy_all, "./datasets/ogbg_molhiv/proxy_all.pt")
            torch.save(proxy_all, path)
    elif method == 'common':
        args = argparse.Namespace(
            cfg_file='configs/personal/common_model_ZINC.yaml',
            mark_done=False, opts=[], repeat=1)
        for n in range(0, repeat):
            model = golden_model_train_ZINC(args)
            # extract given model layers.
            # optimizer.zero_grad()
            for name0, module0 in model.named_children():
                if name0 == 'post_mp':
                    for name1, module1 in module0.named_children():
                        for name2, module2 in module1.named_children():
                            for name3, module3 in module2.named_children():
                                if name3 == '0':
                                    module3.register_forward_hook(hook_fn_forward)
                # elif name0 == 'mp':
                #     for name1, module1 in module0.named_children():
                #         module1.register_forward_hook(hook_fn_forward)
            model.eval()
            # TODO(wby) Here we need to use this model and conduct reference in order to generate proxy vector
            loaders = create_sequential_loader_ZINC()
            indices0 = loaders[0].dataset._indices
            indices1 = loaders[1].dataset._indices
            indices2 = loaders[2].dataset._indices
            full_length = indices0.__len__() + indices1.__len__() + indices2.__len__()
            proxy_all = torch.zeros(full_length, 64, device=torch.device(cfg.device))
            for i in range(0, 3):
                loader = loaders[i]
                j = 0
                for batch in loader:
                    batch.to(torch.device(cfg.device))
                    model(batch)
                    batch_hid = total_feat_out[0].clone()
                    del total_feat_out[0]
                    if j == 0:
                        proxy_loader = batch_hid.clone()  # 注意，这里默认的操作是global_add_pool，所以出来之后的结果和global_mean_pool的结果会有差别！
                    else:
                        proxy_vec = batch_hid.clone()
                        proxy_loader = torch.cat([proxy_loader, proxy_vec], dim=0)
                    j = j + 1
                proxy_all[loaders[i].dataset._indices] = proxy_loader
            name0 = pooling
            name1 = str(n)
            name = 'proxy_all_' + name0 + '_common_ZINC_' + name1 + '.pt'
            path = os.path.join('datasets/ZINC/', name)
            # torch.save(proxy_all, "./datasets/ogbg_molhiv/proxy_all.pt")
            torch.save(proxy_all, path)


def create_bad_vec_ZINC(pooling='add', repeat=1, method='bad'):
    """
    和create_golden_vec类似，都是创建proxy task，区别在于本函数采用了hook函数的方法来提取中间层结果
    :return:
    """
    cfg.model.graph_pooling = pooling
    if method == 'golden':
        args = argparse.Namespace(cfg_file='configs/personal/golden_model_ZINC.yaml', mark_done=False, opts=[], repeat=1)
        for n in range(0, repeat):
            model = golden_model_train_ZINC(args)
            # extract given model layers.
            # optimizer.zero_grad()
            for name0, module0 in model.named_children():
                if name0 == 'post_mp':
                    for name1, module1 in module0.named_children():
                        for name2, module2 in module1.named_children():
                            for name3, module3 in module2.named_children():
                                if name3 == '0':
                                    module3.register_forward_hook(hook_fn_forward)
                # elif name0 == 'mp':
                #     for name1, module1 in module0.named_children():
                #         module1.register_forward_hook(hook_fn_forward)
            model.eval()
            # TODO(wby) Here we need to use this model and conduct reference in order to generate proxy vector
            loaders = create_sequential_loader_ZINC()
            indices = []
            indices.append(loaders[0].dataset.data.y.shape[0])
            indices.append(loaders[1].dataset.data.y.shape[0])
            indices.append(loaders[2].dataset.data.y.shape[0])
            proxy_all = []
            # proxy_all.append(torch.zeros(indices0, 64, device=torch.device(cfg.device))
            for i in range(0, 3):
                loader = loaders[i]
                j = 0
                for batch in loader:
                    batch.to(torch.device(cfg.device))
                    model(batch)
                    batch_hid = total_feat_out[0].clone()
                    del total_feat_out[0]
                    if j == 0:
                        proxy_loader = batch_hid.clone()  # 注意，这里默认的操作是global_add_pool，所以出来之后的结果和global_mean_pool的结果会有差别！
                    else:
                        proxy_vec = batch_hid.clone()
                        proxy_loader = torch.cat([proxy_loader, proxy_vec], dim=0)
                    j = j + 1
                # proxy_all[loaders[i].dataset._indices] =
                proxy_all.append(proxy_loader)
            name0 = pooling
            name1 = str(n)
            name = 'proxy_all_' + name0 + '_golden_ZINC_' + name1 + '.pt'
            path = os.path.join('datasets/ZINC/', name)
            # torch.save(proxy_all, "./datasets/ogbg_molhiv/proxy_all.pt")
            torch.save(proxy_all, path)
    elif method == 'bad':
        args = argparse.Namespace(cfg_file='configs/personal/bad_model_ZINC.yaml', mark_done=False, opts=[], repeat=1)
        for n in range(0, repeat):
            model = golden_model_train_ZINC(args)
            # extract given model layers.
            # optimizer.zero_grad()
            for name0, module0 in model.named_children():
                if name0 == 'post_mp':
                    for name1, module1 in module0.named_children():
                        for name2, module2 in module1.named_children():
                            for name3, module3 in module2.named_children():
                                if name3 == '0':
                                    module3.register_forward_hook(hook_fn_forward)
                # elif name0 == 'mp':
                #     for name1, module1 in module0.named_children():
                #         module1.register_forward_hook(hook_fn_forward)
            model.eval()
            # TODO(wby) Here we need to use this model and conduct reference in order to generate proxy vector
            loaders = create_sequential_loader_ZINC()
            indices = []
            indices.append(loaders[0].dataset.data.y.shape[0])
            indices.append(loaders[1].dataset.data.y.shape[0])
            indices.append(loaders[2].dataset.data.y.shape[0])
            proxy_all = []
            # proxy_all.append(torch.zeros(indices0, 64, device=torch.device(cfg.device))
            for i in range(0, 3):
                loader = loaders[i]
                j = 0
                for batch in loader:
                    batch.to(torch.device(cfg.device))
                    model(batch)
                    batch_hid = total_feat_out[0].clone()
                    del total_feat_out[0]
                    if j == 0:
                        proxy_loader = batch_hid.clone()  # 注意，这里默认的操作是global_add_pool，所以出来之后的结果和global_mean_pool的结果会有差别！
                    else:
                        proxy_vec = batch_hid.clone()
                        proxy_loader = torch.cat([proxy_loader, proxy_vec], dim=0)
                    j = j + 1
                # proxy_all[loaders[i].dataset._indices] =
                proxy_all.append(proxy_loader)
            name0 = pooling
            name1 = str(n)
            name = 'proxy_all_' + name0 + '_bad_ZINC_' + name1 + '.pt'
            path = os.path.join('datasets/ZINC/', name)
            # torch.save(proxy_all, "./datasets/ogbg_molhiv/proxy_all.pt")
            torch.save(proxy_all, path)
    elif method == 'common':
        args = argparse.Namespace(
            cfg_file='configs/personal/common_model_ZINC.yaml',
            mark_done=False, opts=[], repeat=1)
        for n in range(0, repeat):
            model = golden_model_train_ZINC(args)
            # extract given model layers.
            # optimizer.zero_grad()
            for name0, module0 in model.named_children():
                if name0 == 'post_mp':
                    for name1, module1 in module0.named_children():
                        for name2, module2 in module1.named_children():
                            for name3, module3 in module2.named_children():
                                if name3 == '0':
                                    module3.register_forward_hook(hook_fn_forward)
                # elif name0 == 'mp':
                #     for name1, module1 in module0.named_children():
                #         module1.register_forward_hook(hook_fn_forward)
            model.eval()
            # TODO(wby) Here we need to use this model and conduct reference in order to generate proxy vector
            loaders = create_sequential_loader_ZINC()
            indices0 = loaders[0].dataset._indices
            indices1 = loaders[1].dataset._indices
            indices2 = loaders[2].dataset._indices
            full_length = indices0.__len__() + indices1.__len__() + indices2.__len__()
            proxy_all = torch.zeros(full_length, 64, device=torch.device(cfg.device))
            for i in range(0, 3):
                loader = loaders[i]
                j = 0
                for batch in loader:
                    batch.to(torch.device(cfg.device))
                    model(batch)
                    batch_hid = total_feat_out[0].clone()
                    del total_feat_out[0]
                    if j == 0:
                        proxy_loader = batch_hid.clone()  # 注意，这里默认的操作是global_add_pool，所以出来之后的结果和global_mean_pool的结果会有差别！
                    else:
                        proxy_vec = batch_hid.clone()
                        proxy_loader = torch.cat([proxy_loader, proxy_vec], dim=0)
                    j = j + 1
                proxy_all[loaders[i].dataset._indices] = proxy_loader
            name0 = pooling
            name1 = str(n)
            name = 'proxy_all_' + name0 + '_common_ZINC_' + name1 + '.pt'
            path = os.path.join('datasets/ZINC/', name)
            # torch.save(proxy_all, "./datasets/ogbg_molhiv/proxy_all.pt")
            torch.save(proxy_all, path)


def create_random_vec(repeat=1, dim=64, num_graph=41127):
    """
    这个函数的作用是对数据集中的每一张图生成一个[1, dim]维度的proxy vec，每个分量在[0, 1]之间，且均归一化
    """
    for n in range(0, repeat):
        rand_all = F.normalize(torch.rand(num_graph, dim))
        rand_all = rand_all.to(torch.device('cpu'))
        name = './datasets/ogbg_molhiv/rand_all_' + str(n) + '.pt'
        torch.save(rand_all, name)
        print("Success, file saved to %s." % name)


def attach_golden_vec(loaders):
    proxy_all = torch.load("./datasets/ogbg_molhiv/proxy_all_add_golden_0.pt")
    proxy_all.requires_grad_(False)
    proxy_all = proxy_all.to(torch.device('cpu'))
    length = proxy_all.shape[0]
    proxy_all = proxy_all.reshape([proxy_all.numel(), 1])
    slice = torch.linspace(0, length * 64, steps=length+1)
    for i in range(0, loaders.__len__()):
        loader = loaders[i]
        set_dataset_attr_eig(loader.dataset, 'eig_vec', proxy_all, slice)


def attach_eigen_vec(loaders):
    evec_all = torch.load('./datasets/ogbg_molhiv/proxy_all_eigen.pt')
    slice = loaders[0].dataset.slices['x']
    for i in range(0, loaders.__len__()):
        loader = loaders[i]
        set_dataset_attr_eig(loader.dataset, 'eig_vec', evec_all, slice)


def attach_golden_vec2(loaders):
    """
    本函数可以理解为一个临时函数，和attach_golden_vec完全一致，仅仅是在加载proxy vec的路径不一样
    :param loaders:
    :return:
    """
    proxy_all = torch.load("./datasets/ogbg_molhiv/proxy_all_add_golden_0.pt")
    proxy_all.requires_grad_(False)
    proxy_all = proxy_all.to(torch.device('cpu'))
    length = proxy_all.shape[0]
    proxy_all = proxy_all.reshape([proxy_all.numel(), 1])
    slice = torch.linspace(0, length * 64, steps=length + 1)
    for i in range(0, loaders.__len__()):
        loader = loaders[i]
        set_dataset_attr_eig(loader.dataset, 'eig_vec', proxy_all, slice)


def attach_golden_vec_ZINC(loaders):
    proxy_all = torch.load("./datasets/ZINC/proxy_all_add_bad_ZINC_2.pt")
    for i in range(0, loaders.__len__()):
        proxy_all_loader = proxy_all[i]
        proxy_all_loader.requires_grad_(False)
        proxy_all_loader = proxy_all_loader.to(torch.device('cpu'))
        length = proxy_all_loader.shape[0]
        proxy_all_loader = proxy_all_loader.reshape([proxy_all_loader.numel(), 1])
        slice = torch.linspace(0, length * 64, steps=length + 1)
        loader = loaders[i]
        set_dataset_attr_eig(loader.dataset, 'eig_vec', proxy_all_loader, slice)


def attach_random_vec(loaders):
    proxy_all = torch.load("./datasets/ogbg_molhiv/rand_all_4.pt")
    proxy_all.requires_grad_(False)
    proxy_all = proxy_all.to(torch.device('cpu'))
    length = proxy_all.shape[0]
    proxy_all = proxy_all.reshape([proxy_all.numel(), 1])
    slice = torch.linspace(0, length * 64, steps=length+1)
    for i in range(0, loaders.__len__()):
        loader = loaders[i]
        set_dataset_attr_eig(loader.dataset, 'eig_vec', proxy_all, slice)


if __name__ == '__main__':
    method = 'bad'
    if method == 'golden':
        create_golden_vec_ZINC(repeat=10, pooling='add')
    elif method == 'bad':
        create_bad_vec_ZINC(method='bad', repeat=10, pooling='add')
    elif method == 'rand':
        create_random_vec(repeat=10, dim=64, num_graph=41127)
    elif method == 'common':
        create_golden_vec(pooling='add', repeat=1, method='common')