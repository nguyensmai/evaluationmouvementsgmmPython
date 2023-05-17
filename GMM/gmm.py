import pickle
import os
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split


class GaussianMixtureModelPhysicalRehabilitation:
    def __init__(self, data_correct, data_incorrect, number_of_gaussians = [3, 5, 10, 15], random_seeds = [0, 17, 31, 42],
                 training_sizes = [3, 5, 10, 15, 25, 50, 100, 200, 400], validation_sizes = [5, 10, 15, 20, 30, 40, 50],
                 max_iterations = 20, covariance = 'diag', tolerance = 10e-10, num_of_timeframes = 100):
        self.all_correct = data_correct
        self.all_incorrect = data_incorrect
        self.num_of_gaussians = number_of_gaussians
        self.random_seeds = random_seeds
        self.num_of_folds = len(random_seeds)
        self.num_of_timeframes = num_of_timeframes
        self.folds = self.make_splits_cross_validation()
        self.models = []
        self.training_sizes = training_sizes
        self.validation_sizes = validation_sizes
        self.max_iterations = max_iterations
        self.covariance_type = covariance
        self.tol = tolerance
        self.f1_values_validation = []
        self.f1_values_test = []
        self.f1_scores_validation = []
        self.f1_scores_test = []

    # make split for NUM_OF_FOLDS cross validation training
    def make_splits_cross_validation(self):
        folds = []
        all_correct = self.all_correct.reshape((int(len(self.all_correct) / self.num_of_timeframes),
                                                self.num_of_timeframes, len(self.all_correct[0])))
        all_incorrect = self.all_incorrect.reshape((int(len(self.all_incorrect) / self.num_of_timeframes),
                                                  self.num_of_timeframes, len(self.all_incorrect[0])))
        for split_index in range(self.num_of_folds):
            train_correct, test_correct = train_test_split(all_correct, test_size=0.25,
                                                           random_state=self.random_seeds[split_index])
            validation_incorrect, test_incorrect = train_test_split(all_incorrect, test_size=0.25,
                                                           random_state=self.random_seeds[split_index])
            folds.append([train_correct, test_correct, validation_incorrect, test_incorrect])
        return folds

    # train GMM models - for one combination (number of gaussians, training size) train NUM_OF_FOLDS models for each cross validation split
    # in self.models saves 3D matrix of trained models groped first by number of gaussians, then training size and fold number at the end
    def train(self):
        for num_of_gaussians in self.num_of_gaussians:
            models_by_training_sizes = []
            for training_size in self.training_sizes:
                if len(self.folds[0][0]) < training_size + self.validation_sizes[0]:
                    break
                if num_of_gaussians > training_size:
                    continue
                models = []
                for index in range(self.num_of_folds):
                    X = self.folds[index][0][:training_size]
                    X = X.reshape((len(X) * self.num_of_timeframes, len(X[0][0])))
                    models.append(GaussianMixture(n_components=num_of_gaussians, covariance_type=self.covariance_type,
                                                  max_iter=self.max_iterations, tol=self.tol, verbose=2).fit(X))
                    print('Iterations needed: ', models[-1].n_iter_)
                    # print(models[-1].precisions_)
                models_by_training_sizes.append(models)
            self.models.append(models_by_training_sizes)


    # calculate f1 scores and determines thresholds for every trained model with different validation sizes
    # averages scores for one cross validation group
    # in model variables saves values and scores for F1 metric grouped by number of gaussians, training size and validation size
    def evaluate(self):
        for models_by_num_gauss_index in range(len(self.models)):
            f1_values_validation_per_training_size = []
            f1_values_test_per_training_size = []
            f1_scores_validation_per_training_size = []
            f1_scores_test_per_training_size = []
            for models_by_training_size_index in range(len(self.models[models_by_num_gauss_index])):
                training_sizes_to_skip = 0
                training_size_index = 0
                while training_size_index < len(self.training_sizes) and \
                        self.training_sizes[training_size_index] < self.num_of_gaussians[models_by_num_gauss_index]:
                    training_sizes_to_skip += 1
                    training_size_index += 1
                training_size = self.training_sizes[training_sizes_to_skip + models_by_training_size_index]
                f1_values_validation_per_validation_size = []
                f1_values_test_per_validation_size = []
                f1_scores_validation_per_validation_size = []
                f1_scores_test_per_validation_size = []
                for validation_size in self.validation_sizes:
                    values_for_f1_scores_validation_one_group = []
                    values_for_f1_scores_test_one_group = []
                    f1_scores_validation_one_group = []
                    f1_scores_test_one_group = []
                    for split_index in range(self.num_of_folds):
                        if training_size + validation_size > len(self.folds[split_index][0]) or \
                                validation_size > len(self.folds[split_index][2]):
                            break
                        model = self.models[models_by_num_gauss_index][models_by_training_size_index][split_index]
                        positives_validation_set = self.folds[split_index][0][training_size:training_size + validation_size]
                        positives_validation_set = positives_validation_set.reshape((len(positives_validation_set) * self.num_of_timeframes, len(positives_validation_set[0][0])))
                        positives_validation = model.score_samples(positives_validation_set)
                        negatives_validation_set = self.folds[split_index][2][:validation_size]
                        negatives_validation_set = negatives_validation_set.reshape((len(negatives_validation_set) * self.num_of_timeframes,len(negatives_validation_set[0][0])))
                        negatives_validation = model.score_samples(negatives_validation_set)
                        positives_test_set = self.folds[split_index][1]
                        positives_test_set = positives_test_set.reshape((len(positives_test_set) * self.num_of_timeframes, len(positives_test_set[0][0])))
                        positives_test = model.score_samples(positives_test_set)
                        negatives_test_set = self.folds[split_index][3]
                        negatives_test_set = negatives_test_set.reshape((len(negatives_test_set) * self.num_of_timeframes, len(negatives_test_set[0][0])))
                        negatives_test = model.score_samples(negatives_test_set)
                        values_for_thresholds = {}
                        max_f1_score = 0
                        max_f1_score_threshold = 0
                        max_f1_score_values = []
                        total_number = len(positives_validation) + len(negatives_validation)
                        for threshold in np.linspace(min(negatives_validation) - 0.01, max(positives_validation) + 0.01, 10000):
                            tp = sum(i >= threshold for i in positives_validation) * 1.0 / total_number
                            fn = sum(i < threshold for i in positives_validation) * 1.0 / total_number
                            fp = sum(i >= threshold for i in negatives_validation) * 1.0 / total_number
                            tn = sum(i < threshold for i in negatives_validation) * 1.0 / total_number
                            f1_score = tp / (tp + (fp + fn) / 2)
                            values_for_thresholds[threshold] = [[tn, fp], [fn, tp]]
                            if f1_score >= max_f1_score:
                                max_f1_score = f1_score
                                max_f1_score_threshold = threshold
                                max_f1_score_values = [[tn, fp], [fn, tp]]

                        values_for_f1_scores_validation_one_group.append(values_for_thresholds[max_f1_score_threshold])
                        f1_scores_validation_one_group.append(max_f1_score)

                        total_number = len(positives_test) + len(negatives_test)
                        tp = sum(i >= max_f1_score_threshold for i in positives_test) * 1.0 / total_number
                        fn = sum(i < max_f1_score_threshold for i in positives_test) * 1.0 / total_number
                        fp = sum(i >= max_f1_score_threshold for i in negatives_test) * 1.0 / total_number
                        tn = sum(i < max_f1_score_threshold for i in negatives_test) * 1.0 / total_number
                        f1_score = tp / (tp + (fp + fn) / 2)

                        values_for_f1_scores_test_one_group.append([[tn, fp], [fn, tp]])
                        f1_scores_test_one_group.append(f1_score)

                    if len(values_for_f1_scores_validation_one_group) > 0:
                        avg_values_validation = sum(np.array(values_for_f1_scores_validation_one_group)) / self.num_of_folds
                        avg_values_test = sum(np.array(values_for_f1_scores_test_one_group)) / self.num_of_folds
                        avg_f1_score_validation = sum(f1_scores_validation_one_group) / self.num_of_folds
                        avg_f1_score_test = sum(f1_scores_test_one_group) / self.num_of_folds

                        f1_values_validation_per_validation_size.append(avg_values_validation)
                        f1_values_test_per_validation_size.append(avg_values_test)
                        f1_scores_validation_per_validation_size.append(avg_f1_score_validation)
                        f1_scores_test_per_validation_size.append(avg_f1_score_test)

                f1_values_validation_per_training_size.append(f1_values_validation_per_validation_size)
                f1_values_test_per_training_size.append(f1_values_test_per_validation_size)
                f1_scores_validation_per_training_size.append(f1_scores_validation_per_validation_size)
                f1_scores_test_per_training_size.append(f1_scores_test_per_validation_size)

            self.f1_values_validation.append(f1_values_validation_per_training_size)
            self.f1_values_test.append(f1_values_test_per_training_size)
            self.f1_scores_validation.append(f1_scores_validation_per_training_size)
            self.f1_scores_test.append(f1_scores_test_per_training_size)


    def save(self, save_dir, base_name):
        for models_by_num_gauss_index in range(len(self.models)):
            for training_index in range(len(self.models[models_by_num_gauss_index])):
                training_sizes_to_skip = 0
                training_size_index = 0
                while training_size_index < len(self.training_sizes) and \
                        self.training_sizes[training_size_index] < self.num_of_gaussians[models_by_num_gauss_index]:
                    training_sizes_to_skip += 1
                    training_size_index += 1
                for validation_size_index in range(len(self.f1_scores_test[models_by_num_gauss_index][training_index])):
                    output_filename = base_name + '_gauss-' + str(self.num_of_gaussians[models_by_num_gauss_index]) + \
                                      '_maxi-' + str(self.max_iterations) + \
                                      '_covariance-' + self.covariance_type + \
                                      '_trainingSize-' + str(self.training_sizes[training_sizes_to_skip + training_index]) + \
                                      '_validationSize-' + str(self.validation_sizes[validation_size_index])

                    with open(os.path.join(save_dir, output_filename + '.txt'), 'w') as file:
                        file.write('Avg F1 score validation: ' + str(self.f1_scores_validation[models_by_num_gauss_index][training_index][validation_size_index]))
                        file.write('\nAvg F1 values validation :' + str(self.f1_values_validation[models_by_num_gauss_index][training_index][validation_size_index]))
                        file.write('\n\nAvg F1 score test: ' + str(self.f1_scores_test[models_by_num_gauss_index][training_index][validation_size_index]))
                        file.write('\nAvg F1 values test: ' + str(self.f1_values_test[models_by_num_gauss_index][training_index][validation_size_index]))

                    with open(os.path.join(save_dir, output_filename + '_conf_matrix_validation'), 'wb') as file:
                        pickle.dump(self.f1_values_validation[models_by_num_gauss_index][training_index][validation_size_index], file)

                    with open(os.path.join(save_dir, output_filename + '_conf_matrix_test'), 'wb') as file:
                        pickle.dump(self.f1_values_test[models_by_num_gauss_index][training_index][validation_size_index], file)

                    with open(os.path.join(save_dir, output_filename + '_model'), 'wb') as file:
                        pickle.dump(self.models[models_by_num_gauss_index][training_index][0], file)



