from os import listdir, path, getcwd, chdir
from tqdm import tqdm
from scipy.stats import multivariate_normal
import pandas as pd
import numpy as np
import utils
import numba

class ScanpathPrediction:
    def __init__(self, dataset_name, human_scanpaths_dir, dataset_results_dir, models_dir):
        self.models_results = {}
        self.dataset_name   = dataset_name
        self.human_scanpaths_dir = human_scanpaths_dir
        self.dataset_results_dir = dataset_results_dir
        self.models_dir          = models_dir

    def compute_metrics_for_model(self, model_name):
        human_scanpaths_files = listdir(self.human_scanpaths_dir)
        model_output_path     = path.join(dataset_results_dir, model_name)
        # Save current working directory and change it to model's directory
        original_wd = getcwd()
        chdir(path.join(self.models_dir, model_name))
        # Load model's main
        module = importlib.import_module('.main')
        run_visualsearch = getattr(module, 'main')
        for subject in human_scanpaths_files:
            subject_number  = subject[4:6]
            run_visualsearch(self.dataset_name, int(subject_number))

            output_path     = path.join(model_output_path, 'human_subject_' + subject_number)
            human_scanpaths = utils.load_from_json(path.join(output_path, 'Subject_scanpaths.json'))
            for image_name in human_scanpaths:
                scanpath = human_scanpaths[image_name]
                human_fixations_x = np.array(scanpath['X'], dtype=int)
                human_fixations_y = np.array(scanpath['Y'], dtype=int)
                probability_maps_folder = path.join(output_path, path.join('probability_maps', image_name[:-4]))
                for index in range(1, np.size(human_fixations_x)):          
                    probability_map = pd.read_csv(probability_maps_folder + 'fixation_' + str(index))                
                    self.compute_scanpath_prediction_metrics(probability_map, human_fixations_y[:index], human_fixations_x[:index])
        
        # Restore working directory
        chdir(original_wd)

    def compute_scanpath_prediction_metrics(self, probability_map, human_fixations_y, human_fixations_x):
        probability_map = probability_map.to_numpy(dtype=np.float)
        probability_map = normalize(probability_map)
        baseline_map    = center_gaussian(probability_map.shape)

        roc = np.mean(AUCs(probability_map, human_fixations_y, human_fixations_x)) # ¿Promediamos?
        nss = NSS(probability_map, human_fixations_y, human_fixations_x)
        ig  = infogain(probability_map, baseline_map, human_fixations_y, human_fixations_x)

        #en que formato guardar los resultados? json?

    def center_gaussian(self, shape):
        sigma  = [[1, 0], [0, 1]]
        mean   = [shape[0] // 2, shape[1] // 2]
        x_range = np.linspace(0, shape[0], shape[0])
        y_range = np.linspace(0, shape[1], shape[1])

        x_matrix, y_matrix = np.meshgrid(y_range, x_range)
        quantiles = np.transpose([y_matrix.flatten(), x_matrix.flatten()])
        mvn = multivariate_normal.pdf(quantiles, mean=mean, cov=sigma)
        mvn = np.reshape(mvn_at_fixation, shape)

        return mvn

    def normalize(self, probability_map):
        normalized_probability_map = probability_map - np.min(probability_map)
        normalized_probability_map = normalized_probability_map / np.max(normalized_probability_map)

        return normalized_probability_map

    def NSS(self, saliency_map, ground_truth_fixations_y, ground_truth_fixations_x):
        mean = np.mean(saliency_map)
        std  = np.std(saliency_map)
        value = np.copy(saliency_map[ground_truth_fixations_y, ground_truth_fixations_x])
        value -= mean

        if std:
            value /= std

        return value

    def infogain(self, s_map, baseline_map, ground_truth_fixations_y, ground_truth_fixations_x):
        eps = 2.2204e-16

        s_map        = s_map / (np.sum(s_map) * 1.0)
        baseline_map = baseline_map / (np.sum(baseline_map) * 1.0)

        temp = []

        for i in zip(ground_truth_fixations_y, ground_truth_fixations_x):
            temp.append(np.log2(eps + s_map[i[1], i[0]]) - np.log2(eps + baseline_map[i[1], i[0]]))

        return np.mean(temp)

    def AUCs(self, probability_map, ground_truth_fixations_y, ground_truth_fixations_x):
        """ Calculate AUC scores for fixations """
        rocs_per_fixation = []

        for i in tqdm(range(len(ground_truth_fixations_x)), total=len(ground_truth_fixations_x)):
            positive  = probability_map[ground_truth_fixations_y[i], ground_truth_fixations_x[i]]
            negatives = probability_map.flatten()

            this_roc = auc_for_one_positive(positive, negatives)
            rocs_per_fixation.append(this_roc)

        return np.asarray(rocs_per_fixation)

    def auc_for_one_positive(self, positive, negatives):
        """ Computes the AUC score of one single positive sample agains many negatives.
        The result is equal to general_roc([positive], negatives)[0], but computes much
        faster because one can save sorting the negatives.
        """
        return _auc_for_one_positive(positive, np.asarray(negatives))

@numba.jit(nopython=True)
def fill_fixation_map(fixation_map, fixations):
    """fixationmap: 2d array. fixations: Nx2 array of y, x positions"""
    for i in range(len(fixations)):
        fixation_y, fixation_x = fixations[i]
        fixation_map[int(fixation_y), int(fixation_x)] += 1

@numba.jit(nopython=True)
def _auc_for_one_positive(positive, negatives):
    """ Computes the AUC score of one single positive sample agains many negatives.
    The result is equal to general_roc([positive], negatives)[0], but computes much
    faster because one can save sorting the negatives.
    """
    count = 0
    for negative in negatives:
        if negative < positive:
            count += 1
        elif negative == positive:
            count += 0.5

    return count / len(negatives)