import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import csv


def proxy_groundtruth_analysis():
    name0s = ['val_bestepoch', 'test_bestepoch', 'test_best','test','val','val_best']
    name1s = ['proxy51_grid_proxy_mod216_2'] # write groundtruth ranking results
    name2s = ['personal_graph4_grid_proxy_mod216_1', 'personal_graph4_grid_proxy_mod216_2',
              'personal_graph4_grid_proxy_mod216_3', 'personal_graph4_grid_proxy_mod216_4',
              'personal_graph4_grid_proxy_mod216_5']  # write proxy ranking results
    f = open('results/proxy_groundtruth_analysis.csv', 'w')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["name0", "name1", "name2", 'rho', 'tau'])
    for name0 in name0s:
        for name1 in name1s:
            for name2 in name2s:
                path1 = os.path.join('results/', name1, 'agg/', name0 + '.csv')
                csvData1 = pd.read_csv(path1)
                csvData1.sort_values(["mse"], axis=0, ascending=[True], inplace=True)
                x = csvData1.serial.values
                x = pd.Series(x)  # groundtruth ranking

                path2 = os.path.join('results/', name2, 'agg/', name0 + '.csv')
                csvData2 = pd.read_csv(path2)
                csvData2.sort_values(["auc"], axis=0, ascending=[False], inplace=True)
                y = csvData2.serial.values
                y = pd.Series(y)  # proxy ranking

                value = x.argsort().values  # re-sort indices
                sorted_x = pd.Series(value[x - 1])
                sorted_y = pd.Series(value[y - 1])

                rho = sorted_x.corr(sorted_y, method="spearman")  # compute spearman's rho
                # rho = x.corr(y, method='spearman')
                tau = sorted_x.corr(sorted_y, method="kendall")  # compute kendall's tau
                # tau = x.corr(y, method="kendall")
                print('spearman\'s rho = %f' % rho)
                print('kendall\'s tau = %f' % tau)
                title = 'rho = %.4f, tau = %.4f' % (rho, tau)
                fig, ax = plt.subplots()
                # plt.scatter(x, y)
                plt.scatter(sorted_x, sorted_y)
                plt.title(title)
                fig_name = './fig_new/' + name0 + '=' + name1 + '=' + name2
                plt.savefig(fig_name)
                plt.show()  # draw scatter graph
                csv_writer.writerow([name0, name1, name2, rho, tau])
    f.close()


def proxy_groundtruth_analysis_ZINC():
    name0s = ['val_bestepoch', 'test_bestepoch', 'personal_test_val_best', 'test_best', 'test','val', 'val_best']
    name1s = ['proxy_ZINC_grid_proxy_mod216_1', 'proxy_ZINC_grid_proxy_mod216_2', 'proxy_ZINC_grid_proxy_mod216_3']
    name2s = ['personal_graph_ZINC_grid_proxy_mod216_1', 'personal_graph_ZINC_grid_proxy_mod216_2']
    f = open('results/proxy_groundtruth_analysis.csv', 'w')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["name0", "name1", "name2", 'rho', 'tau'])
    for name0 in name0s:
        for name1 in name1s:
            for name2 in name2s:
                path1 = os.path.join('results/', name1, 'agg/', name0 + '.csv')
                csvData1 = pd.read_csv(path1)
                csvData1.sort_values(["mse"], axis=0, ascending=[True], inplace=True)
                x = csvData1.serial.values
                x = pd.Series(x)  # proxy ranking

                path2 = os.path.join('results/', name2, 'agg/', name0 + '.csv')
                csvData2 = pd.read_csv(path2)
                csvData2.sort_values(["mae"], axis=0, ascending=[True], inplace=True)
                y = csvData2.serial.values
                y = pd.Series(y)  # groundtruth ranking

                value = x.argsort().values  # re-sort indices
                sorted_x = pd.Series(value[x - 1])
                sorted_y = pd.Series(value[y - 1])

                rho = sorted_x.corr(sorted_y, method="spearman")  # compute spearman's rho
                # rho = x.corr(y, method='spearman')
                tau = sorted_x.corr(sorted_y, method="kendall")  # compute kendall's tau
                # tau = x.corr(y, method="kendall")
                print('spearman\'s rho = %f' % rho)
                print('kendall\'s tau = %f' % tau)
                title = 'rho = %.4f, tau = %.4f' % (rho, tau)
                fig, ax = plt.subplots()
                # plt.scatter(x, y)
                plt.scatter(sorted_x, sorted_y)
                plt.title(title)
                fig_name = './fig_new/' + name0 + '=' + name1 + '=' + name2
                plt.savefig(fig_name)
                plt.show()  # draw scatter graph
                csv_writer.writerow([name0, name1, name2, rho, tau])
    f.close()


def similarity_analysis():
    name0s = ['val_bestepoch', 'test_bestepoch', 'test_best','test','val']
    name1s = ['proxy31_laplacian_grid_proxy_mod216_1', 'proxy31_laplacian_grid_proxy_mod216_2']
    metric = "mse"  # "mse", "auc"
    if metric == "mse":
        f = open('results/proxy_similarity_analysis.csv', 'w')
    elif metric == ("auc" or "mae" or "loss"):
        f = open('results/groundtruth_similarity_analysis.csv', 'w')
    else:
        f = open('results/ZINC_groundtruth_similarity_analysis.csv', 'w')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["name0", "name1", "name2", 'rho', 'tau'])
    if metric == ("mse" or "mae" or "loss"):
        ascending = True
    else:
        ascending = False
    length = name1s.__len__()
    for name0 in name0s:
        for i in range(0, length):
            for j in range(i+1, length):
                name1 = name1s[i]
                name2 = name1s[j]
                if name1 != name2:
                    path1 = os.path.join('results/', name1, 'agg/', name0 + '.csv')
                    csvData1 = pd.read_csv(path1)
                    csvData1.sort_values([metric], axis=0, ascending=[ascending], inplace=True)
                    x = csvData1.serial.values
                    x = pd.Series(x)  # groundtruth ranking

                    path2 = os.path.join('results/', name2, 'agg/', name0 + '.csv')
                    csvData2 = pd.read_csv(path2)
                    csvData2.sort_values([metric], axis=0, ascending=[ascending], inplace=True)
                    y = csvData2.serial.values
                    y = pd.Series(y)  # proxy ranking

                    value = x.argsort().values  # re-sort indices
                    sorted_x = pd.Series(value[x - 1])
                    sorted_y = pd.Series(value[y - 1])

                    rho = sorted_x.corr(sorted_y, method="spearman")  # compute spearman's rho
                    # rho = x.corr(y, method='spearman')
                    tau = sorted_x.corr(sorted_y, method="kendall")  # compute kendall's tau
                    # tau = x.corr(y, method="kendall")
                    print('spearman\'s rho = %f' % rho)
                    print('kendall\'s tau = %f' % tau)
                    title = 'rho = %.4f, tau = %.4f' % (rho, tau)
                    fig, ax = plt.subplots()
                    # plt.scatter(x, y)
                    plt.scatter(sorted_x, sorted_y)
                    plt.title(title)
                    if metric == "mse":
                        fig_name = './figures/proxy_similarity_fig/' + name0 + '=' + name1 + '=' + name2 + '.png'
                    elif metric == 'auc':
                        fig_name = './figures/groundtruth_similarity_fig/' + name0 + '=' + name1 + '=' + name2 + '.png'
                    else:
                        fig_name = './figures/ZINC_groundtruth_similarity_fig/' + name0 + '=' + name1 + '=' + name2 + '.png'
                    plt.savefig(fig_name)
                    plt.show()  # draw scatter graph
                    csv_writer.writerow([name0, name1, name2, rho, tau])
    f.close()


def similarity_analysis_epoch():
    metric = "mse"  # "mse", "auc"
    name1s = ['proxy51_grid_proxy_mod48_5', 'proxy52_grid_proxy_mod48_5',
              'proxy51_grid_proxy_mod48_6', 'proxy52_grid_proxy_mod48_6', 'proxy53_grid_proxy_mod48_6',
              'proxy51_grid_proxy_mod48_7', 'proxy52_grid_proxy_mod48_7', 'proxy53_grid_proxy_mod48_7',
              'proxy51_grid_proxy_mod48_8', 'proxy52_grid_proxy_mod48_8', 'proxy53_grid_proxy_mod48_8',
              'proxy51_grid_proxy_mod48_9', 'proxy52_grid_proxy_mod48_9']

    length = name1s.__len__()
    for i in range(0, length):
        for j in range(i+1, length):
            name1 = name1s[i]
            name2 = name1s[j]
            if metric == "mse":
                csvname = 'results/csvfile/proxy_epoch' + name1 + '=' + name2 + '.csv'
                f = open(csvname, 'w')
            else:
                f = open('results/groundtruth_epoch_similarity_analysis.csv', 'w')
            csv_writer = csv.writer(f)
            csv_writer.writerow(["epoch", "name1", "name2", 'rho', 'tau'])
            for epoch in range(0, 80):
                file_name = 'testepoch' + str(epoch) + '.csv'
                file_path1 = '/results/' + name1 + '/agg_epoch_proxy'
                file_path2 = '/results/' + name2 + '/agg_epoch_proxy'
                if metric == "mse":
                    ascending = True
                else:
                    ascending = False
                path1 = os.path.join(file_path1, file_name)
                csvData1 = pd.read_csv(path1)
                csvData1.sort_values([metric], axis=0, ascending=[ascending], inplace=True)
                x = csvData1.serial.values
                x = pd.Series(x)  # groundtruth ranking

                path2 = os.path.join(file_path2, file_name)
                csvData2 = pd.read_csv(path2)
                csvData2.sort_values([metric], axis=0, ascending=[ascending], inplace=True)
                y = csvData2.serial.values
                y = pd.Series(y)  # proxy ranking

                value = x.argsort().values  # re-sort indices
                sorted_x = pd.Series(value[x - 1])
                sorted_y = pd.Series(value[y - 1])

                rho = sorted_x.corr(sorted_y, method="spearman")  # compute spearman's rho
                # rho = x.corr(y, method='spearman')
                tau = sorted_x.corr(sorted_y, method="kendall")  # compute kendall's tau
                # tau = x.corr(y, method="kendall")
                csv_writer.writerow([epoch, name1, name2, rho, tau])
            print('spearman\'s rho = %f' % rho)
            print('kendall\'s tau = %f' % tau)
            f.close()
    for i in range(0, length):
        for j in range(i+1, length):
            name1 = name1s[i]
            name2 = name1s[j]
            if metric == "mse":
                csvname = 'results/csvfile/proxy_epoch' + name1 + '=' + name2 + '.csv'
                csvData = pd.read_csv(csvname)
                x = csvData['epoch']
                y = csvData['rho']
                title = name1 + ' vs. ' + name2
                # title = 'rho = %.4f, tau = %.4f' % (rho, tau)
                fig, ax = plt.subplots()
                # # plt.scatter(x, y)
                plt.plot(x, y, 'c.-', linewidth=1)
                plt.title(title)
                plt.xlim(0, 80)
                plt.ylim(0, 1)
                if metric == "mse":
                    fig_name = './figures/proxy_similarity_epoch_fig/' + name1 + '=' + name2 + '.png'
                else:
                    fig_name = './figures/groundtruth_similarity_epoch_fig/' + name1 + '=' + name2 + '.png'
                plt.savefig(fig_name)
                plt.show()  # draw scatter graph


def proxy_groundtruth_analysis_epoch():
    name0s = ['val_best']
    # , 'test_bestepoch', 'personal_test_val_best', 'test_best', 'test','val','val_best']
    # name1s = ['proxy51_grid_proxy_mod48_9', 'proxy52_grid_proxy_mod48_9', 'proxy53_grid_proxy_mod48_9']
    name1s = ['proxy51_grid_proxy_mod216_2']
    # name2s = ['personal_graph4_grid_proxy_mod48_1', 'personal_graph4_grid_proxy_mod48_2',
    #           'personal_graph4_grid_proxy_mod48_3', 'personal_graph4_grid_proxy_mod48_4', 'personal_graph4_grid_proxy_mod48_5', 'personal_graph4_grid_proxy_mod48_6']
    # name2s = ['personal_graph4_grid_proxy_mod216_1', 'personal_graph4_grid_proxy_mod216_2',
    name2s= ['personal_graph4_grid_proxy_mod216_3', 'personal_graph4_grid_proxy_mod216_4']
              # 'personal_graph4_grid_proxy_mod216_5']
    name3s = ['test','val']
    metric = "mse" # proxy task ranking metric
    length = name2s.__len__()
    for name0 in name0s:
        for name1 in name1s: #proxy
            for name2 in name2s: #groundtruth
                for name3 in name3s: #test/val
                    csvname = 'results/csvfile/proxy_groundtruth_analysis_epoch/' + name3 + '=' + name0 + "=" + name1 + "=" + name2 + '.csv'
                    f = open(csvname, 'w')
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(["epoch", "metric_best", "split", "name1",  "name2", "rho", "tau"])
                    for epoch in range(0, 80):
                        file_name1 = name3 + 'epoch' + str(epoch) + '.csv'
                        file_name2 = name0 + '.csv'
                        file_path1 = '/results/' + name1 + '/agg_epoch_proxy'
                        file_path2 = '/results/' + name2 + '/agg'
                        if metric == 'mse':
                            ascending = True
                        else:
                            ascending = False
                        path1 = os.path.join(file_path1, file_name1)
                        path2 = os.path.join(file_path2, file_name2)
                        csvData1 = pd.read_csv(path1)
                        csvData1.sort_values([metric], axis=0, ascending=[ascending], inplace=True)
                        x = csvData1.serial.values
                        x = pd.Series(x)  # groundtruth ranking

                        path2 = os.path.join(file_path2, file_name2)
                        csvData2 = pd.read_csv(path2)
                        csvData2.sort_values(["auc"], axis=0, ascending=False, inplace=True)
                        y = csvData2.serial.values
                        y = pd.Series(y)  # proxy ranking
                        value = x.argsort().values  # re-sort indices
                        sorted_x = pd.Series(value[x - 1])
                        sorted_y = pd.Series(value[y - 1])

                        rho = sorted_x.corr(sorted_y, method="spearman")  # compute spearman's rho
                        # rho = x.corr(y, method='spearman')
                        tau = sorted_x.corr(sorted_y, method="kendall")  # compute kendall's tau
                        # tau = x.corr(y, method="kendall")
                        csv_writer.writerow([epoch, name0, name3, name1, name2, rho, tau])
                    print('spearman\'s rho = %f' % rho)
                    print('kendall\'s tau = %f' % tau)
                    f.close()
                    csvData = pd.read_csv(csvname)
                    x = csvData['epoch']
                    y = csvData['rho']
                    title = name1 + 'vs.' + name2 + '/' + name3 + '/' + name0
                    fig, ax = plt.subplots()
                    plt.plot(x, y, 'c.-', linewidth=1)
                    plt.title(title)
                    plt.xlim(0, 80)
                    plt.ylim(-1, 1)
                    fig_name = './figures/proxy_groundtruth_epoch_fig/' + name0 + '=' + name1 + '=' + name2 + '=' + name3 + '.png'
                    plt.savefig(fig_name)
                    plt.show()


def proxy_groundtruth_analysis_epoch_ZINC():
    name0s = ['val_best', 'test_bestepoch', 'personal_test_val_best', 'test_best', 'test','val','val_best']
    # name1s = ['proxy51_grid_proxy_mod48_9', 'proxy52_grid_proxy_mod48_9', 'proxy53_grid_proxy_mod48_9']
    name1s = ['proxy_ZINC_grid_proxy_mod216_1','proxy_ZINC_grid_proxy_mod216_2','proxy_ZINC_grid_proxy_mod216_3']
    # name2s = ['personal_graph4_grid_proxy_mod48_1', 'personal_graph4_grid_proxy_mod48_2',
    #           'personal_graph4_grid_proxy_mod48_3', 'personal_graph4_grid_proxy_mod48_4', 'personal_graph4_grid_proxy_mod48_5', 'personal_graph4_grid_proxy_mod48_6']
    # name2s = ['personal_graph4_grid_proxy_mod216_1', 'personal_graph4_grid_proxy_mod216_2',
    name2s= ['personal_graph_ZINC_grid_proxy_mod216_1', 'personal_graph_ZINC_grid_proxy_mod216_2']
              # 'personal_graph4_grid_proxy_mod216_5']
    name3s = ['test','val']
    metric = "mse" # proxy task ranking metric
    length = name2s.__len__()
    for name0 in name0s:
        for name1 in name1s: #proxy
            for name2 in name2s: #groundtruth
                for name3 in name3s: #test/val
                    csvname = 'results/csvfile/proxy_groundtruth_analysis_epoch/' + name3 + '=' + name0 + "=" + name1 + "=" + name2 + '.csv'
                    f = open(csvname, 'w')
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(["epoch", "metric_best", "split", "name1",  "name2", "rho", "tau"])
                    for epoch in range(0, 80):
                        file_name1 = name3 + 'epoch' + str(epoch) + '.csv'
                        file_name2 = name0 + '.csv'
                        file_path1 = '/results/' + name1 + '/agg_epoch_proxy'
                        file_path2 = '/results/' + name2 + '/agg'
                        if metric == 'mse':
                            ascending = True
                        else:
                            ascending = False
                        path1 = os.path.join(file_path1, file_name1)
                        csvData1 = pd.read_csv(path1)
                        csvData1.sort_values([metric], axis=0, ascending=[ascending], inplace=True)
                        x = csvData1.serial.values
                        x = pd.Series(x)  # groundtruth ranking

                        path2 = os.path.join(file_path2, file_name2)
                        csvData2 = pd.read_csv(path2)
                        csvData2.sort_values(["mae"], axis=0, ascending=True, inplace=True)
                        y = csvData2.serial.values
                        y = pd.Series(y)  # proxy ranking
                        value = x.argsort().values  # re-sort indices
                        sorted_x = pd.Series(value[x - 1])
                        sorted_y = pd.Series(value[y - 1])

                        rho = sorted_x.corr(sorted_y, method="spearman")  # compute spearman's rho
                        # rho = x.corr(y, method='spearman')
                        tau = sorted_x.corr(sorted_y, method="kendall")  # compute kendall's tau
                        # tau = x.corr(y, method="kendall")
                        csv_writer.writerow([epoch, name0, name3, name1, name2, rho, tau])
                    print('spearman\'s rho = %f' % rho)
                    print('kendall\'s tau = %f' % tau)
                    f.close()
                    csvData = pd.read_csv(csvname)
                    x = csvData['epoch']
                    y = csvData['rho']
                    title = name1 + 'vs.' + name2 + '/' + name3 + '/' + name0
                    fig, ax = plt.subplots()
                    plt.plot(x, y, 'c.-', linewidth=1)
                    plt.title(title)
                    plt.xlim(0, 80)
                    plt.ylim(0, 0.5)
                    fig_name = './figures/proxy_groundtruth_epoch_fig/' + name0 + '=' + name1 + '=' + name2 + '=' + name3 + '.png'
                    plt.savefig(fig_name)
                    plt.show()


if __name__ == '__main__':
    proxy_groundtruth_analysis_epoch_ZINC()
    # proxy_groundtruth_analysis()
    # similarity_analysis()
    # similarity_analysis_epoch()
