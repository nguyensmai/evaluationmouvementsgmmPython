import os
import pickle
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt


# function that produces confusion matrices
def make_conf_matrix(src_dir, dst_dir, groups, exercises, data_types):
    number_of_rows = len(exercises)
    number_of_columns = len(data_types)
    files = os.listdir(src_dir)
    for group in groups:

        sns.set(rc={'figure.figsize': (8 * number_of_columns + 4, 8 * number_of_rows)})
        fig, ax = plt.subplots(number_of_rows, number_of_columns)
        counter = 0

        for exercise, data_type in product(exercises, data_types):
            conf_matrix_id = group + '_' + data_type + '_' + exercise
            for file in files:
                if conf_matrix_id in file and 'onf_matrix' in file and 'valid' in file and 'trainingSize-50' in file:
                    f = open(os.path.join(src_dir, file), 'rb')
                    cf_matrix = pickle.load(f)
                    f.close()
                    column = counter % number_of_columns
                    row = int(counter / number_of_columns)
                    cbar = False if column != number_of_columns - 1 else True
                    sns.heatmap(cf_matrix, ax=ax[row][column], annot=True,
                                cbar=cbar, cmap='Blues', annot_kws={'fontsize': 48, 'fontweight': 'bold'}, linewidths=1,
                                linecolor='black')
                    ax[row][column].xaxis.set_ticklabels(['False', 'True'], fontsize=32)
                    ax[row][column].yaxis.set_ticklabels(['False', 'True'], fontsize=32)
                    if column == 0:
                        ax[row][column].set_ylabel(exercise, fontsize=48, fontweight='bold')
                    if row == 0:
                        ax[row][column].set_title(data_type, fontsize=48, fontweight='bold')
                    counter += 1
                    break

        output_plot_path = os.path.join(dst_dir, group + '.png')
        plt.savefig(output_plot_path)


if __name__ == '__main__':
    groups = ['group1A2A3']
    conf_matrices_src_dir = 'E:\PhD\GCN for PR paper + evaluation\code\GCN-for-PR-evaluation\\results GMM\\'
    conf_matrices_dst_dir = 'G:\Datasets\Keraal dataset organized new\conf matrices GMM'
    exercises = ['CTK', 'ELK', 'RTK']
    data_types = ['kinect', 'openpose', 'blazepose']
    make_conf_matrix(conf_matrices_src_dir, conf_matrices_dst_dir, groups, exercises, data_types)
