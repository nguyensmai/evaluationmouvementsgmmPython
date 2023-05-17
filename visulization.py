import os
import pickle
from itertools import product
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from numpy import exp, arange
from pylab import meshgrid, cm, imshow, contour, clabel, colorbar, axis, title, show
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.autolayout"] = True


def make_plot_f1_acc_over_train_size(plots, output_plot_path1):
    fig, ax = plt.subplots()
    if type(plots) is dict:
        train_sizes = []
        accuracies = []
        f1_scores = []
        for train_size, _ in plots.items():
            train_sizes.append(train_size)
            accuracies.append(plots[train_size][0])
            f1_scores.append(plots[train_size][1])
    else:
        train_sizes = plots[0]
        accuracies = plots[1]
        f1_scores = plots[2]
        ax.plot(train_sizes, accuracies, linestyle='--', marker='o')


    ax.plot(train_sizes, accuracies, linestyle='--', marker='o', label='Accuracy')
    ax.plot(train_sizes, f1_scores, linestyle='--', marker='o', label='F1 score')
    ax.legend()
    plt.legend()
    plt.ylim(0.5, 1.0)
    plt.xlabel('Training sizes')

    plt.title(' '.join(output_plot_path1.split('\\')[-1][:-4].split('_')))
    plt.savefig(output_plot_path1)


def make_plot_f1_over_train_valid_size(plots, output_plot_path1, output_plot_path2):
    fig, ax = plt.subplots()
    for val_size, value in plots.items():
        if type(value) is dict:
            train_sizes = []
            accuracies = []
            for train_size, _ in plots[val_size].items():
                train_sizes.append(train_size)
                accuracies.append(plots[val_size][train_size][0])
        else:
            train_sizes = value[0]
            accuracies = value[1]
        ax.plot(train_sizes, accuracies, linestyle='--', marker='o')
    ax.legend()
    plt.ylim(0.5, 1.0)
    plt.xlabel('Training sizes')

    plt.title(' '.join(output_plot_path1.split('\\')[-1][:-4].split('_')))
    plt.savefig(output_plot_path1)

    fig, ax = plt.subplots()
    for val_size, value in plots.items():
        if type(value) is dict:
            train_sizes = []
            f1_scores = []  # [plots[val_size][train_size][0] for train_size in train_dict.keys()]
            for train_size, _ in plots[val_size].items():
                train_sizes.append(train_size)
                f1_scores.append(plots[val_size][train_size][1])
        else:
            train_sizes = value[0]
            f1_scores = value[2]
        ax.plot(train_sizes, f1_scores, linestyle='--', marker='o', label=('val_size ' + str(val_size)))
    ax.legend()
    plt.ylim(0.5, 1.0)
    plt.xlabel('Training sizes')
    plt.title(' '.join(output_plot_path2.split('\\')[-1][:-4].split('_')))
    plt.legend()
    plt.savefig(output_plot_path2)


def make_plots_kimore_gmm(conf_matrices_src_dir, conf_matrices_dst_dir, number_of_gaussians, exercises, training_sizes,
                          validation_sizes):
    files = os.listdir(conf_matrices_src_dir)
    for num_of_gauss in number_of_gaussians:
        scores_by_valid_size_avg = {}
        for exercise in exercises:
            scores_by_valid_size = {}
            for validation_size, training_size in product(validation_sizes, training_sizes):
                found = False
                for file in files:
                    if ('gauss-' + str(num_of_gauss)) in file and \
                            exercise in file and \
                            ('trainingSize-' + str(training_size) + '_') in file and \
                            ('validationSize-' + str(validation_size) + '_') in file and \
                            'conf_matrix_test' in file:
                        found = True
                        break
                if not found:
                    continue
                if validation_size not in scores_by_valid_size:
                    scores_by_valid_size[validation_size] = [[], [], []]
                scores_by_valid_size[validation_size][0].append(training_size)
                f = open(os.path.join(conf_matrices_src_dir, file), 'rb')
                [[tn, fp], [fn, tp]] = pickle.load(f)
                f.close()
                acc = tp + tn
                f1_score = tp / (tp + (fp + fn) / 2)
                scores_by_valid_size[validation_size][1].append(acc)
                scores_by_valid_size[validation_size][2].append(f1_score)

            name = conf_matrices_dst_dir + exercise + '_gauss-' + str(num_of_gauss)
            make_plot_f1_over_train_valid_size(scores_by_valid_size_avg, name + '_acc_test.png',
                                               name + '_f1_score_test.png')

            for val_size in scores_by_valid_size.keys():
                if val_size not in scores_by_valid_size_avg:
                    scores_by_valid_size_avg[val_size] = {}
                for ind in range(len(scores_by_valid_size[val_size][0])):
                    train_size = scores_by_valid_size[val_size][0][ind]
                    if train_size not in scores_by_valid_size_avg[val_size]:
                        scores_by_valid_size_avg[val_size][train_size] = [0, 0, 0]
                    scores_by_valid_size_avg[val_size][train_size][0] += scores_by_valid_size[val_size][1][ind]
                    scores_by_valid_size_avg[val_size][train_size][1] += scores_by_valid_size[val_size][2][ind]
                    scores_by_valid_size_avg[val_size][train_size][2] += 1

        for val_size in scores_by_valid_size_avg.keys():
            for train_size in scores_by_valid_size_avg[val_size].keys():
                scores_by_valid_size_avg[val_size][train_size][0] = scores_by_valid_size_avg[val_size][train_size][0] / \
                                                                    scores_by_valid_size_avg[val_size][train_size][2]
                scores_by_valid_size_avg[val_size][train_size][1] = scores_by_valid_size_avg[val_size][train_size][1] / \
                                                                    scores_by_valid_size_avg[val_size][train_size][2]

        make_plot_f1_over_train_valid_size(scores_by_valid_size_avg, conf_matrices_dst_dir + 'Kimore_acc_avg_test.png',
                                           conf_matrices_dst_dir + 'Kimore_f1_score_avg_test.png')


def make_plots_keraal_gmm(conf_matrices_src_dir, conf_matrices_dst_dir, number_of_gaussians, exercises, training_sizes,
                          validation_sizes):
    files = os.listdir(conf_matrices_src_dir)
    for num_of_gauss in number_of_gaussians:
        for group, data_type in product(groups, data_types):
            scores_by_valid_size_avg = {}
            for exercise in exercises:
                scores_by_valid_size = {}
                for validation_size, training_size in product(validation_sizes, training_sizes):
                    found = False
                    for file in files:
                        if ('gauss-' + str(num_of_gauss)) in file and \
                                group in file and \
                                data_type in file and \
                                exercise in file and \
                                ('trainingSize-' + str(training_size) + '_') in file and \
                                ('validationSize-' + str(validation_size) + '_') in file and \
                                'conf_matrix_valid' in file:
                            found = True
                            break
                    if not found:
                        continue
                    if validation_size not in scores_by_valid_size:
                        scores_by_valid_size[validation_size] = [[], [], []]
                    scores_by_valid_size[validation_size][0].append(training_size)
                    f = open(os.path.join(conf_matrices_src_dir, file), 'rb')
                    [[tn, fp], [fn, tp]] = pickle.load(f)
                    f.close()
                    acc = tp + tn
                    f1_score = tp / (tp + (fp + fn) / 2)
                    scores_by_valid_size[validation_size][1].append(acc)
                    scores_by_valid_size[validation_size][2].append(f1_score)

                name = conf_matrices_dst_dir + 'Keraal_' + group + '_' + data_type + '_' + exercise + '_gauss-' + str(num_of_gauss)
                make_plot_f1_over_train_valid_size(scores_by_valid_size_avg, name + '_acc_test.png',
                                                   name + '_f1_score_test.png')

                for val_size in scores_by_valid_size.keys():
                    if val_size not in scores_by_valid_size_avg:
                        scores_by_valid_size_avg[val_size] = {}
                    for ind in range(len(scores_by_valid_size[val_size][0])):
                        train_size = scores_by_valid_size[val_size][0][ind]
                        if train_size not in scores_by_valid_size_avg[val_size]:
                            scores_by_valid_size_avg[val_size][train_size] = [0, 0, 0]
                        scores_by_valid_size_avg[val_size][train_size][0] += scores_by_valid_size[val_size][1][ind]
                        scores_by_valid_size_avg[val_size][train_size][1] += scores_by_valid_size[val_size][2][ind]
                        scores_by_valid_size_avg[val_size][train_size][2] += 1



            for val_size in scores_by_valid_size_avg.keys():
                for train_size in scores_by_valid_size_avg[val_size].keys():
                    scores_by_valid_size_avg[val_size][train_size][0] = scores_by_valid_size_avg[val_size][train_size][0] / \
                                                                        scores_by_valid_size_avg[val_size][train_size][2]
                    scores_by_valid_size_avg[val_size][train_size][1] = scores_by_valid_size_avg[val_size][train_size][1] / \
                                                                        scores_by_valid_size_avg[val_size][train_size][2]


            name =  conf_matrices_dst_dir + 'Keraal_' + group + '_' + data_type + '_gauss-' + str(num_of_gauss)
            make_plot_f1_over_train_valid_size(scores_by_valid_size_avg, name + '_acc_avg_test.png',
                                               name + '_f1_score_avg_test.png')


def make_plots_keraal_gcn(conf_matrices_src_dir, conf_matrices_dst_dir, exercises, training_sizes):
    files = os.listdir(conf_matrices_src_dir)
    for group, data_type in product(groups, data_types):
        scores_by_train_size_avg = {}
        for exercise in exercises:
            scores_by_train_size = [[], [], []]
            for training_size in training_sizes:
                found = False
                for file in files:
                    if group in file and \
                            data_type in file and \
                            exercise in file and \
                            ('ts-' + str(training_size) + '_') in file and \
                            'conf_matrix' in file:
                        found = True
                        break
                if not found:
                    continue
                scores_by_train_size[0].append(training_size)
                f = open(os.path.join(conf_matrices_src_dir, file), 'rb')
                [[tn, fp], [fn, tp]] = pickle.load(f)
                f.close()
                acc = tp + tn
                f1_score = tp / (tp + (fp + fn) / 2)
                scores_by_train_size[1].append(acc)
                scores_by_train_size[2].append(f1_score)

            name = conf_matrices_dst_dir + 'Keraal_' + group + '_' + data_type + '_' + exercise
            make_plot_f1_acc_over_train_size(scores_by_train_size, name + '_f1_acc_test.png')

            for train_size in scores_by_train_size[0]:
                if train_size not in scores_by_train_size_avg:
                    scores_by_train_size_avg[train_size] = [0, 0, 0]
            for ind in range(len(scores_by_train_size[0])):
                scores_by_train_size_avg[scores_by_train_size[0][ind]][0] += scores_by_train_size[1][ind]
                scores_by_train_size_avg[scores_by_train_size[0][ind]][1] += scores_by_train_size[2][ind]
                scores_by_train_size_avg[scores_by_train_size[0][ind]][2] += 1

        for train_size in scores_by_train_size_avg.keys():
            scores_by_train_size_avg[train_size][0] = scores_by_train_size_avg[train_size][0] / \
                                                                    scores_by_train_size_avg[train_size][2]
            scores_by_train_size_avg[train_size][1] = scores_by_train_size_avg[train_size][1] / \
                                                                    scores_by_train_size_avg[train_size][2]

        name =  conf_matrices_dst_dir + 'Keraal_' + group + '_' + data_type
        make_plot_f1_acc_over_train_size(scores_by_train_size_avg, name + '_acc_f1_avg_test.png')


def make_plots_kimore_gcn(conf_matrices_src_dir, conf_matrices_dst_dir, exercises, training_sizes):
    files = os.listdir(conf_matrices_src_dir)
    scores_by_train_size_avg = {}
    for exercise in exercises:
        scores_by_train_size = [[], [], []]
        for training_size in training_sizes:
            found = False
            for file in files:
                if exercise in file and \
                        ('ts-' + str(training_size) + '_') in file and \
                        'conf_matrix' in file:
                    found = True
                    break
            if not found:
                continue
            scores_by_train_size[0].append(training_size)
            f = open(os.path.join(conf_matrices_src_dir, file), 'rb')
            [[tn, fp], [fn, tp]] = pickle.load(f)
            f.close()
            acc = tp + tn
            f1_score = tp / (tp + (fp + fn) / 2)
            scores_by_train_size[1].append(acc)
            scores_by_train_size[2].append(f1_score)

        name = conf_matrices_dst_dir + exercise
        make_plot_f1_acc_over_train_size(scores_by_train_size, name + '_f1_acc_test.png')

        for train_size in scores_by_train_size[0]:
            if train_size not in scores_by_train_size_avg:
                scores_by_train_size_avg[train_size] = [0, 0, 0]
        for ind in range(len(scores_by_train_size[0])):
            scores_by_train_size_avg[scores_by_train_size[0][ind]][0] += scores_by_train_size[1][ind]
            scores_by_train_size_avg[scores_by_train_size[0][ind]][1] += scores_by_train_size[2][ind]
            scores_by_train_size_avg[scores_by_train_size[0][ind]][2] += 1

    for train_size in scores_by_train_size_avg.keys():
        scores_by_train_size_avg[train_size][0] = scores_by_train_size_avg[train_size][0] / \
                                                  scores_by_train_size_avg[train_size][2]
        scores_by_train_size_avg[train_size][1] = scores_by_train_size_avg[train_size][1] / \
                                                  scores_by_train_size_avg[train_size][2]

    make_plot_f1_acc_over_train_size(scores_by_train_size_avg, conf_matrices_dst_dir + 'Kimore_acc_f1_avg_test.png')


if __name__ == '__main__':
    groups = ['group1A', 'group1A2A', 'group3']
    conf_matrices_src_dir = 'E:\PhD\GCN for PR paper + evaluation\code\GCN-for-PR-evaluation\\results\\kimore\\'
    conf_matrices_dst_dir = 'G:\Datasets\Keraal dataset organized new\conf matrices GCN\\'
    exercises = ['CTK', 'ELK', 'RTK']
    exercises = ['Kimore_ex1', 'Kimore_ex2', 'Kimore_ex3', 'Kimore_ex4', 'Kimore_ex5']
    data_types = ['kinect', 'openpose', 'blazepose']
    training_sizes = [3, 5, 10, 15, 25, 50, 100, 200, 300]
    validation_sizes = [5, 10, 15, 20, 30, 40, 50, 100]
    number_of_gaussians = [5, 10, 15]
    # make_plots_keraal_gmm(conf_matrices_src_dir, conf_matrices_dst_dir, number_of_gaussians, exercises, training_sizes,
    #                       validation_sizes)
    # make_plots_kimore_gmm(conf_matrices_src_dir, conf_matrices_dst_dir, number_of_gaussians, exercises, training_sizes,
    #                       validation_sizes)
    # make_plots_keraal_gcn(conf_matrices_src_dir, conf_matrices_dst_dir, exercises, training_sizes)
    make_plots_kimore_gcn(conf_matrices_src_dir, conf_matrices_dst_dir, exercises, training_sizes)