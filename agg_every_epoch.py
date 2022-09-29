import logging
import argparse
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.utils.io import (dict_list_to_json,
                                               dict_list_to_tb, dict_to_json,
                                               json_to_dict_list,
                                               makedirs,
                                               string_to_python)

def name_to_dict(run):
    cols = run.split('-')[1:]
    keys, vals = [], []
    for col in cols:
        try:
            key, val = col.split('=')
        except Exception:
            print(col)
        keys.append(key)
        vals.append(string_to_python(val))
    return dict(zip(keys, vals))


def rm_keys(dict, keys):
    for key in keys:
        dict.pop(key, None)


def agg_batch_epoch(dir, metric_best='auto', epoch=0):
    '''
    Aggregate across results from multiple experiments via grid search

    Args:
        dir (str): Directory of the results, containing multiple experiments
        metric_best (str, optional): The metric for selecting the best
        validation performance. Options: auto, accuracy, auc.

    '''
    import pandas as pd
    results = {'test': []}
    list_dir = os.listdir(dir)
    list_dir.sort()
    i = 0
    for run in list_dir:  # TODO(wby) listdir is an arbitrary order
        if run != 'agg':
            dict_name = name_to_dict(run)
            dir_run = os.path.join(dir, run, 'agg')
            if os.path.isdir(dir_run):
                i = i + 1  # in order to give the right serial number for each combination (self-defined)
                split = 'test'  # split = 'val' 'val' 'train'
                dir_split = os.path.join(dir_run, split)
                fname_stats = os.path.join(dir_split, 'stats.json')
                dict_stats = json_to_dict_list(fname_stats)[epoch]  # get best epoch
                rm_keys(dict_stats,
                        ['lr', 'lr_std', 'eta', 'eta_std', 'params_std'])  # remove these attribution
                serial = {'serial': i}
                results[split].append({**serial, **dict_name, **dict_stats})  # 将组合方式和训练结果进行组合，前面参数为组合方式，后面参数为训练结果
    dir_out = os.path.join(dir, 'agg_epoch')
    # makedirs_rm_exist(dir_out)
    for key in results:
        if len(results[key]) > 0:
            results[key] = pd.DataFrame(results[key])
            results[key] = results[key].sort_values(
                list(serial.keys()) + list(dict_name.keys()), ascending=[True] * (len(serial) + len(dict_name)))
            fname = os.path.join(dir_out, 'testepoch' + str(epoch) + '.csv')      # 将每个combination的best.json存入csv当中
            results[key].to_csv(fname, index=False)

    print('Results aggregated across models saved in {}'.format(dir_out))


def agg_batch_proxy_epoch(dir, metric_best='auto', maxepoch=80):  # TODO(wby) may still have some problems, please refer to agg_batch
    import pandas as pd
    for epoch in range(0, maxepoch):
        cfg.metric_agg = 'argmin'
        results = {'val': []}
        list_dir = os.listdir(dir)
        list_dir.sort()
        i = 0

        for run in list_dir:
            if run != 'agg':
                dict_name = name_to_dict(run)
                dir_run = os.path.join(dir, run, 'agg')  # 找到相对应的单个configuration的agg
                if os.path.isdir(dir_run):
                    i = i + 1
                    split = 'val'
                    dir_split = os.path.join(dir_run, split)
                    fname_stats = os.path.join(dir_split, 'stats.json')
                    dict_stats = json_to_dict_list(fname_stats)[epoch]  # get best val epoch
                    serial = {'serial': i}
                    rm_keys(dict_stats,
                            ['lr', 'lr_std', 'eta', 'eta_std', 'params_std'])
                    results[split].append({**serial, **dict_name, **dict_stats})
        dir_out = os.path.join(dir, 'agg_epoch_proxy')
        makedirs(dir_out)
        for key in results:
            if len(results[key]) > 0:
                results[key] = pd.DataFrame(results[key])
                results[key] = results[key].sort_values(
                    list(serial.keys()) + list(dict_name.keys()), ascending=[True] * (len(serial) + len(dict_name)))
                fname = os.path.join(dir_out, 'valepoch' + str(epoch) + '.csv')
                results[key].to_csv(fname, index=False)
        print('Results aggregated across models saved in {}'.format(dir_out))


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(
        description='Train a classification model')
    parser.add_argument('--dir', dest='dir', help='Dir for batch of results',
                        required=True, type=str)
    parser.add_argument('--metric', dest='metric',
                        help='metric to select best epoch', required=False,
                        type=str, default='auto')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    agg_batch_proxy_epoch(args.dir, args.metric, maxepoch=80)