import logging
import os
import numpy as np

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.utils.io import (dict_list_to_json,
                                               dict_list_to_tb, dict_to_json,
                                               json_to_dict_list,
                                               makedirs_rm_exist,
                                               string_to_python)

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


def is_seed(s):
    try:
        int(s)
        return True
    except Exception:
        return False


def is_split(s):
    if s in ['train', 'val', 'test']:
        return True
    else:
        return False


def join_list(l1, l2):
    assert len(l1) == len(l2), \
        'Results with different seeds must have the save format'
    for i in range(len(l1)):
        l1[i] += l2[i]
    return l1


def agg_dict_list(dict_list):
    """
    Aggregate a list of dictionaries: mean + std
    Args:
        dict_list: list of dictionaries

    """
    dict_agg = {'epoch': dict_list[0]['epoch']}
    for key in dict_list[0]:
        if key != 'epoch':
            value = np.array([dict[key] for dict in dict_list])
            dict_agg[key] = np.mean(value).round(cfg.round)  # 这里round是指保留几位小数！
            dict_agg['{}_std'.format(key)] = np.std(value).round(cfg.round)
    return dict_agg


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


def agg_runs(dir, metric_best='auto'):
    r'''
    Aggregate over different random seeds of a single experiment

    Args:
        dir (str): Directory of the results, containing 1 experiment
        metric_best (str, optional): The metric for selecting the best
        validation performance. Options: auto, accuracy, auc.

    '''
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
        results_best[key] = agg_dict_list(results_best[key]) # TODO (wby) 这里是对每一个repeat的最好值取平均吗？查看一下agg_dict_list的用法！对于epoch的值是怎么取平均的？
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


def agg_batch(dir, metric_best='auto'):
    '''
    Aggregate across results from multiple experiments via grid search

    Args:
        dir (str): Directory of the results, containing multiple experiments
        metric_best (str, optional): The metric for selecting the best
        validation performance. Options: auto, accuracy, auc.

    '''
    import pandas as pd
    results = {'train': [], 'val': [], 'test': []}
    list_dir = os.listdir(dir)
    list_dir.sort()
    i = 0
    for run in list_dir:  # TODO(wby) listdir is an arbitrary order
        if run != 'agg':
            dict_name = name_to_dict(run)
            dir_run = os.path.join(dir, run, 'agg')
            if os.path.isdir(dir_run):
                i = i + 1  # in order to give the right serial number for each combination (self-defined)
                for split in os.listdir(dir_run):  # split = 'test' 'val' 'train'
                    dir_split = os.path.join(dir_run, split)
                    fname_stats = os.path.join(dir_split, 'best.json')
                    dict_stats = json_to_dict_list(fname_stats)[
                        -1]  # get best epoch
                    rm_keys(dict_stats,
                            ['lr', 'lr_std', 'eta', 'eta_std', 'params_std'])  # remove these attribution
                    serial = {'serial': i}
                    results[split].append({**serial, **dict_name, **dict_stats})  # 将组合方式和训练结果进行组合，前面参数为组合方式，后面参数为训练结果
    dir_out = os.path.join(dir, 'agg')
    makedirs_rm_exist(dir_out)
    for key in results:
        if len(results[key]) > 0:
            results[key] = pd.DataFrame(results[key])
            results[key] = results[key].sort_values(
                list(serial.keys()) + list(dict_name.keys()), ascending=[True] * (len(serial) + len(dict_name)))
            fname = os.path.join(dir_out, '{}_best.csv'.format(key))      # 将每个combination的best.json存入csv当中
            results[key].to_csv(fname, index=False)

    results = {'train': [], 'val': [], 'test': []}
    list_dir = os.listdir(dir)
    list_dir.sort()
    i = 0
    for run in list_dir:
        if run != 'agg':
            dict_name = name_to_dict(run)
            dir_run = os.path.join(dir, run, 'agg')
            if os.path.isdir(dir_run):
                i = i + 1
                for split in os.listdir(dir_run):
                    dir_split = os.path.join(dir_run, split)
                    fname_stats = os.path.join(dir_split, 'stats.json')
                    dict_stats = json_to_dict_list(fname_stats)[
                        -1]  # get last epoch
                    rm_keys(dict_stats,
                            ['lr', 'lr_std', 'eta', 'eta_std', 'params_std'])
                    serial = {'serial': i}
                    results[split].append({**serial, **dict_name, **dict_stats})
    dir_out = os.path.join(dir, 'agg')
    for key in results:
        if len(results[key]) > 0:
            results[key] = pd.DataFrame(results[key])
            results[key] = results[key].sort_values(
                list(serial.keys()) + list(dict_name.keys()), ascending=[True] * (len(serial) + len(dict_name)))
            fname = os.path.join(dir_out, '{}.csv'.format(key))  # 三次平均之后最后一个epoch
            results[key].to_csv(fname, index=False)

    results = {'train': [], 'val': [], 'test': []}
    list_dir = os.listdir(dir)
    list_dir.sort()
    i = 0
    for run in list_dir:
        if run != 'agg':
            dict_name = name_to_dict(run)
            dir_run = os.path.join(dir, run, 'agg')
            if os.path.isdir(dir_run):
                i = i + 1
                for split in os.listdir(dir_run):
                    dir_split = os.path.join(dir_run, split)
                    fname_stats = os.path.join(dir_split, 'stats.json')
                    dict_stats = json_to_dict_list(
                        fname_stats)  # get best epoch
                    if metric_best == 'auto':
                        metric = 'auc' if 'auc' in dict_stats[0] \
                            else 'accuracy'
                    else:
                        metric = metric_best
                    performance_np = np.array(  # noqa
                        [stats[metric] for stats in dict_stats])
                    dict_stats = dict_stats[eval("performance_np.{}()".format(
                        cfg.metric_agg))]
                    serial = {'serial': i}
                    rm_keys(dict_stats,
                            ['lr', 'lr_std', 'eta', 'eta_std', 'params_std'])
                    results[split].append({**serial, **dict_name, **dict_stats})
    dir_out = os.path.join(dir, 'agg')
    for key in results:
        if len(results[key]) > 0:
            results[key] = pd.DataFrame(results[key])
            results[key] = results[key].sort_values(
                list(serial.keys()) + list(dict_name.keys()), ascending=[True] * (len(serial) + len(dict_name)))
            fname = os.path.join(dir_out, '{}_bestepoch.csv'.format(key))  # 三次平均之后得到的最好epoch
            results[key].to_csv(fname, index=False)

    results = {'train': [], 'val': [], 'test': []}
    list_dir = os.listdir(dir)
    list_dir.sort()
    i = 0
    for run in list_dir:
        if run != 'agg':
            dict_name = name_to_dict(run)
            dir_run = os.path.join(dir, run, 'agg')
            if os.path.isdir(dir_run):
                i = i + 1
                split = 'val'
                dir_split = os.path.join(dir_run, split)
                fname_stats = os.path.join(dir_split, 'best.json')
                dict_stats = json_to_dict_list(fname_stats)[
                    -1]  # get best val epoch
                serial = {'serial': i}
                best_val_epoch = dict_stats['epoch']
                split = 'test'
                dir_split = os.path.join(dir_run, split)
                fname_stats = os.path.join(dir_split, 'stats.json')
                dict_stats = json_to_dict_list(fname_stats)[best_val_epoch]
                rm_keys(dict_stats,
                        ['lr', 'lr_std', 'eta', 'eta_std', 'params_std'])
                results[split].append({**serial, **dict_name, **dict_stats})
    dir_out = os.path.join(dir, 'agg')
    for key in results:
        if len(results[key]) > 0:
            results[key] = pd.DataFrame(results[key])
            results[key] = results[key].sort_values(
                list(serial.keys()) + list(dict_name.keys()), ascending=[True] * (len(serial) + len(dict_name)))
            fname = os.path.join(dir_out, 'personal_test_val_best.csv')
            results[key].to_csv(fname, index=False)

    print('Results aggregated across models saved in {}'.format(dir_out))


def agg_batch_proxy(dir, metric_best='auto'):  # TODO(wby) may still have some problems, please refer to agg_batch
    import pandas as pd
    cfg.metric_agg = 'argmin'
    results = {'train': [], 'val': [], 'test': []}
    list_dir = os.listdir(dir)
    list_dir.sort()
    i = 0
    for run in list_dir:
        if run != 'agg':
            dict_name = name_to_dict(run)
            dir_run = os.path.join(dir, run, 'agg')  # 找到相对应的单个configuration的agg
            if os.path.isdir(dir_run):
                i = i + 1
                for split in os.listdir(dir_run):  # test, val, train
                    dir_split = os.path.join(dir_run, split)
                    fname_stats = os.path.join(dir_split, 'best.json')
                    dict_stats = json_to_dict_list(fname_stats)[
                        -1]  # get best val epoch
                    serial = {'serial': i}
                    rm_keys(dict_stats,
                            ['lr', 'lr_std', 'eta', 'eta_std', 'params_std'])
                    results[split].append({**serial, **dict_name, **dict_stats})
    dir_out = os.path.join(dir, 'agg')
    makedirs_rm_exist(dir_out)
    for key in results:
        if len(results[key]) > 0:
            results[key] = pd.DataFrame(results[key])
            results[key] = results[key].sort_values(
                list(serial.keys()) + list(dict_name.keys()), ascending=[True] * (len(serial) + len(dict_name)))
            fname = os.path.join(dir_out, '{}_best.csv'.format(key))
            results[key].to_csv(fname, index=False)

    results = {'train': [], 'val': [], 'test': []}
    list_dir = os.listdir(dir)
    list_dir.sort()
    i = 0
    for run in list_dir:
        if run != 'agg':
            dict_name = name_to_dict(run)
            dir_run = os.path.join(dir, run, 'agg')
            if os.path.isdir(dir_run):
                i = i + 1
                for split in os.listdir(dir_run):
                    dir_split = os.path.join(dir_run, split)
                    fname_stats = os.path.join(dir_split, 'stats.json')
                    dict_stats = json_to_dict_list(fname_stats)[
                        -1]  # get last epoch
                    serial = {'serial': i}
                    rm_keys(dict_stats,
                            ['lr', 'lr_std', 'eta', 'eta_std', 'params_std'])
                    results[split].append({**serial, **dict_name, **dict_stats})
    dir_out = os.path.join(dir, 'agg')
    for key in results:
        if len(results[key]) > 0:
            results[key] = pd.DataFrame(results[key])
            results[key] = results[key].sort_values(
                list(serial.keys()) + list(dict_name.keys()), ascending=[True] * (len(serial) + len(dict_name)))
            fname = os.path.join(dir_out, '{}.csv'.format(key))
            results[key].to_csv(fname, index=False)

    results = {'train': [], 'val': [], 'test': []}
    list_dir = os.listdir(dir)
    list_dir.sort()
    i = 0
    for run in list_dir:
        if run != 'agg':
            dict_name = name_to_dict(run)
            dir_run = os.path.join(dir, run, 'agg')
            if os.path.isdir(dir_run):
                i = i + 1
                for split in os.listdir(dir_run):
                    dir_split = os.path.join(dir_run, split)
                    fname_stats = os.path.join(dir_split, 'stats.json')
                    dict_stats = json_to_dict_list(
                        fname_stats)  # get best epoch
                    if metric_best == 'auto':
                        metric = 'mse' if 'mse' in dict_stats[0] \
                            else 'mae'
                    else:
                        metric = metric_best
                    performance_np = np.array(  # noqa
                        [stats[metric] for stats in dict_stats])
                    dict_stats = dict_stats[eval("performance_np.{}()".format(
                        cfg.metric_agg))]
                    serial = {'serial': i}
                    rm_keys(dict_stats,
                            ['lr', 'lr_std', 'eta', 'eta_std', 'params_std'])
                    results[split].append({**serial, **dict_name, **dict_stats})
    dir_out = os.path.join(dir, 'agg')
    for key in results:
        if len(results[key]) > 0:
            results[key] = pd.DataFrame(results[key])
            results[key] = results[key].sort_values(
                list(serial.keys()) + list(dict_name.keys()), ascending=[True] * (len(serial) + len(dict_name)))
            fname = os.path.join(dir_out, '{}_bestepoch.csv'.format(key))
            results[key].to_csv(fname, index=False)
# ===================================self-defined test-val-bestepoch============================
    results = {'train': [], 'val': [], 'test': []}
    list_dir = os.listdir(dir)
    list_dir.sort()
    i = 0
    for run in list_dir:
        if run != 'agg':
            dict_name = name_to_dict(run)
            dir_run = os.path.join(dir, run, 'agg')
            if os.path.isdir(dir_run):
                i = i + 1
                split = 'val'
                dir_split = os.path.join(dir_run, split)
                fname_stats = os.path.join(dir_split, 'best.json')
                dict_stats = json_to_dict_list(fname_stats)[
                    -1]  # get best val epoch
                serial = {'serial': i}
                best_val_epoch = dict_stats['epoch']
                split = 'test'
                dir_split = os.path.join(dir_run, split)
                fname_stats = os.path.join(dir_split, 'stats.json')
                dict_stats = json_to_dict_list(fname_stats)[best_val_epoch]
                rm_keys(dict_stats,
                        ['lr', 'lr_std', 'eta', 'eta_std', 'params_std'])
                results[split].append({**serial, **dict_name, **dict_stats})
    dir_out = os.path.join(dir, 'agg')
    for key in results:
        if len(results[key]) > 0:
            results[key] = pd.DataFrame(results[key])
            results[key] = results[key].sort_values(
                list(serial.keys()) + list(dict_name.keys()), ascending=[True] * (len(serial) + len(dict_name)))
            fname = os.path.join(dir_out, 'personal_test_val_best.csv')
            results[key].to_csv(fname, index=False)

    print('Results aggregated across models saved in {}'.format(dir_out))


def agg_runs_inference(dir, metric_best='auto'):
    r'''
    Aggregate over different random seeds of a single experiment

    Args:
        dir (str): Directory of the results, containing 1 experiment
        metric_best (str, optional): The metric for selecting the best
        validation performance. Options: auto, accuracy, auc.

    '''
    results = {'test': None}
    results_best = {'test': None}
    for seed in os.listdir(dir):
        if is_seed(seed):
            dir_seed = os.path.join(dir, seed)
            best_epoch = 0
            split = 'test'
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
        results_best[key] = agg_dict_list(results_best[key]) # TODO (wby) 这里是对每一个repeat的最好值取平均吗？查看一下agg_dict_list的用法！对于epoch的值是怎么取平均的？
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


def agg_batch_inference(dir, metric_best='auto'):
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
                split = 'test'
                dir_split = os.path.join(dir_run, split)
                fname_stats = os.path.join(dir_split, 'best.json')
                dict_stats = json_to_dict_list(fname_stats)[
                    -1]  # get best epoch
                rm_keys(dict_stats,
                        ['lr', 'lr_std', 'eta', 'eta_std', 'params_std'])  # remove these attribution
                serial = {'serial': i}
                results[split].append({**serial, **dict_name, **dict_stats})  # 将组合方式和训练结果进行组合，前面参数为组合方式，后面参数为训练结果
    dir_out = os.path.join(dir, 'agg')
    makedirs_rm_exist(dir_out)
    for key in results:
        if len(results[key]) > 0:
            results[key] = pd.DataFrame(results[key])
            results[key] = results[key].sort_values(
                list(serial.keys()) + list(dict_name.keys()), ascending=[True] * (len(serial) + len(dict_name)))
            fname = os.path.join(dir_out, '{}_best.csv'.format(key))      # 将每个combination的best.json存入csv当中
            results[key].to_csv(fname, index=False)

    results = {'test': []}
    list_dir = os.listdir(dir)
    list_dir.sort()
    i = 0
    for run in list_dir:
        if run != 'agg':
            dict_name = name_to_dict(run)
            dir_run = os.path.join(dir, run, 'agg')
            if os.path.isdir(dir_run):
                i = i + 1
                split = 'test'
                dir_split = os.path.join(dir_run, split)
                fname_stats = os.path.join(dir_split, 'stats.json')
                dict_stats = json_to_dict_list(fname_stats)[
                    -1]  # get last epoch
                rm_keys(dict_stats,
                        ['lr', 'lr_std', 'eta', 'eta_std', 'params_std'])
                serial = {'serial': i}
                results[split].append({**serial, **dict_name, **dict_stats})
    dir_out = os.path.join(dir, 'agg')
    for key in results:
        if len(results[key]) > 0:
            results[key] = pd.DataFrame(results[key])
            results[key] = results[key].sort_values(
                list(serial.keys()) + list(dict_name.keys()), ascending=[True] * (len(serial) + len(dict_name)))
            fname = os.path.join(dir_out, '{}.csv'.format(key))  # 三次平均之后最后一个epoch
            results[key].to_csv(fname, index=False)

    print('Results aggregated across models saved in {}'.format(dir_out))


def agg_batch_proxy_inference(dir, metric_best='auto'):  # TODO(wby) may still have some problems, please refer to agg_batch
    import pandas as pd
    cfg.metric_agg = 'argmin'
    results = {'test': []}
    list_dir = os.listdir(dir)
    list_dir.sort()
    i = 0
    for run in list_dir:
        if run != 'agg':
            dict_name = name_to_dict(run)
            dir_run = os.path.join(dir, run, 'agg')  # 找到相对应的单个configuration的agg
            if os.path.isdir(dir_run):
                i = i + 1
                split = 'test'
                dir_split = os.path.join(dir_run, split)
                fname_stats = os.path.join(dir_split, 'best.json')
                dict_stats = json_to_dict_list(fname_stats)[
                    -1]  # get best val epoch
                serial = {'serial': i}
                rm_keys(dict_stats,
                        ['lr', 'lr_std', 'eta', 'eta_std', 'params_std'])
                results[split].append({**serial, **dict_name, **dict_stats})
    dir_out = os.path.join(dir, 'agg')
    makedirs_rm_exist(dir_out)
    for key in results:
        if len(results[key]) > 0:
            results[key] = pd.DataFrame(results[key])
            results[key] = results[key].sort_values(
                list(serial.keys()) + list(dict_name.keys()), ascending=[True] * (len(serial) + len(dict_name)))
            fname = os.path.join(dir_out, '{}_best.csv'.format(key))
            results[key].to_csv(fname, index=False)

    results = {'test': []}
    list_dir = os.listdir(dir)
    list_dir.sort()
    i = 0
    for run in list_dir:
        if run != 'agg':
            dict_name = name_to_dict(run)
            dir_run = os.path.join(dir, run, 'agg')
            if os.path.isdir(dir_run):
                i = i + 1
                split = 'test'
                dir_split = os.path.join(dir_run, split)
                fname_stats = os.path.join(dir_split, 'stats.json')
                dict_stats = json_to_dict_list(fname_stats)[
                    -1]  # get last epoch
                serial = {'serial': i}
                rm_keys(dict_stats,
                        ['lr', 'lr_std', 'eta', 'eta_std', 'params_std'])
                results[split].append({**serial, **dict_name, **dict_stats})
    dir_out = os.path.join(dir, 'agg')
    for key in results:
        if len(results[key]) > 0:
            results[key] = pd.DataFrame(results[key])
            results[key] = results[key].sort_values(
                list(serial.keys()) + list(dict_name.keys()), ascending=[True] * (len(serial) + len(dict_name)))
            fname = os.path.join(dir_out, '{}.csv'.format(key))
            results[key].to_csv(fname, index=False)

    print('Results aggregated across models saved in {}'.format(dir_out))
