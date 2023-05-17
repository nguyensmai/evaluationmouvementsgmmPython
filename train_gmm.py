import numpy as np
from GMM.gmm import GaussianMixtureModelPhysicalRehabilitation
from itertools import product


if __name__ == '__main__':
    groups = ['group1A', 'group1A2A', 'group1A2A3', 'group3']
    exercises = ['CTK', 'ELK', 'RTK']
    data_types = ['kinect', 'openpose', 'blazepose']
    number_of_gaussians = [3, 5, 10, 15]
    random_seeds = [0, 17, 31, 42]
    training_sizes = [3, 5, 10, 15, 25, 50, 100, 200, 400]
    validation_sizes = [5, 10, 15, 20, 30, 40, 50]
    max_iterations = 150
    covariance = 'full'
    tolerance = 1e-3
    output_dir = 'E:\PhD\GCN for PR paper + evaluation\code\GCN-for-PR-evaluation\\results GMM 2\\'
    input_dir = 'G:\Datasets\Keraal dataset organized new\gmm_interpolated\\'
    for data_type, group, exercise in product(data_types, groups, exercises):
        all_correct = np.load(input_dir + data_type + '_' + group + '_' + exercise + '_correct.npy')
        all_incorrect = np.load(input_dir + data_type + '_' + group + '_' + exercise + '_incorrect.npy')
        gmm = GaussianMixtureModelPhysicalRehabilitation(all_correct, all_incorrect,
                                                         number_of_gaussians=number_of_gaussians,
                                                         random_seeds=random_seeds,
                                                         training_sizes=training_sizes,
                                                         validation_sizes=validation_sizes,
                                                         max_iterations=max_iterations,
                                                         covariance=covariance,
                                                         tolerance=tolerance)
        gmm.train()
        gmm.evaluate()
        gmm.save(output_dir, group + '_' + data_type + '_' + exercise)