from torch_geometric.graphgym.utils.agg_runs import (agg_runs,
                                                     agg_batch_proxy,
                                                     agg_batch,
                                                     is_split,
                                                     join_list,
                                                     agg_dict_list,
                                                     is_seed,
                                                     json_to_dict_list,
                                                     dict_list_to_json,
                                                     makedirs_rm_exist,
                                                     dict_list_to_tb,
                                                     dict_to_json)
from torch_geometric.graphgym.config import cfg
import os
import logging
import numpy as np
import pandas as pd
try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None

def agg_new_metric(dir, metric_best='auto', metric_agg='argmax'):
    cfg.metric_agg = metric_agg
    results = {'train': None, 'val': None, 'test': None}
    results_best = {'train': None, 'val': None, 'test': None}
    for seed in os.listdir(dir):
        if is_seed(seed):
            dir_seed = os.path.join(dir, seed)
            split = 'val'
            if split in os.listdir(dir_seed):  # search best epoch
                dir_split = os.path.join(dir_seed, split)
                fname_stats = os.path.join(dir_split, 'stats.json')
                stats_list = json_to_dict_list(fname_stats)
                if metric_best == 'auto':
                    metric = 'auc' if 'auc' in stats_list[0] else 'accuracy'
                else:
                    metric = metric_best
                performance_np = np.array(  # noqa
                    [stats[metric] for stats in stats_list])
                best_epoch = \
                    stats_list[
                        eval("performance_np.{}()".format(cfg.metric_agg))][
                        'epoch']
                print(best_epoch)  # 根据val得到的bestepoch

            for split in os.listdir(dir_seed):
                if is_split(split):
                    dir_split = os.path.join(dir_seed, split)
                    fname_stats = os.path.join(dir_split, 'stats.json')
                    stats_list = json_to_dict_list(fname_stats)
                    stats_best = [
                        stats for stats in stats_list
                        if stats['epoch'] == best_epoch
                    ][0]
                    print(stats_best)
                    stats_list = [[stats] for stats in stats_list]
                    if results[split] is None:
                        results[split] = stats_list
                    else:
                        results[split] = join_list(results[split], stats_list)
                    if results_best[split] is None:
                        results_best[split] = [stats_best]
                    else:
                        results_best[split] += [stats_best]
    results = {k: v for k, v in results.items() if v is not None}  # rm None
    results_best = {k: v
                    for k, v in results_best.items()
                    if v is not None}  # remove None
    for key in results:
        for i in range(len(results[key])):
            results[key][i] = agg_dict_list(results[key][i])
    for key in results_best:
        results_best[key] = agg_dict_list(
            results_best[key])  # TODO (wby) 这里是对每一个repeat的最好值取平均吗？查看一下agg_dict_list的用法！对于epoch的值是怎么取平均的？
    # save aggregated results
    for key, value in results.items():
        dir_out = os.path.join(dir, 'agg', key)
        makedirs_rm_exist(dir_out)
        fname = os.path.join(dir_out, 'stats.json')
        dict_list_to_json(value, fname)

        if cfg.tensorboard_agg:
            if SummaryWriter is None:
                raise ImportError(
                    'Tensorboard support requires `tensorboardX`.')
            writer = SummaryWriter(dir_out)
            dict_list_to_tb(value, writer)
            writer.close()
    for key, value in results_best.items():
        dir_out = os.path.join(dir, 'agg', key)
        fname = os.path.join(dir_out, 'best.json')
        dict_to_json(value, fname)
    logging.info('Results aggregated across runs saved in {}'.format(
        os.path.join(dir, 'agg')))


if __name__ == '__main__':
    new_metric = 'loss'
    dir = 'results/personal_graph4_grid_proxy_mod216_1_loss'
    task = 'proxy'
    list_dir = os.listdir(dir)
    list_dir.sort()
    i = 0
    if task == 'groundtruth':
        for run in list_dir:
            if (run != 'agg') and (run != 'config.yaml'):
                run_dir = dir + '/' + run
                agg_new_metric(run_dir, metric_best=new_metric, metric_agg='argmax')
        agg_batch(dir, new_metric)
    elif task == 'proxy':
        for run in list_dir:
            if (run != 'agg') and (run != 'config.yaml'):
                run_dir = dir + '/' + run
                agg_new_metric(run_dir, metric_best=new_metric, metric_agg='argmin')
        agg_batch_proxy(dir, new_metric)




