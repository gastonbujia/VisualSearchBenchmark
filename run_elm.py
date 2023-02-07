#import scripts.loader as loader
#import scripts.constants as constants
from Models.nnIBS import visualsearch
from Models.nnIBS.scripts import loader, constants
from os import path
import argparse
import utils
import constants as global_constants
import Metrics.main as metrics_module

" Runs visualsearch/main.py according to the supplied parameters "

def setup_and_run(dataset_name, config_name, image_name, image_range, human_subject, number_of_processes, save_probability_maps):
    dataset_path = path.join(constants.DATASETS_PATH, dataset_name)
    output_path  = path.join(constants.RESULTS_PATH, path.join(dataset_name + '_dataset', 'nnIBS'))

    trials_properties_file = path.join(dataset_path, 'trials_properties.json')

    dataset_info      = loader.load_dataset_info(dataset_path)
    output_path       = loader.create_output_folders(output_path, config_name, image_name, image_range, human_subject)
    checkpoint        = loader.load_checkpoint(output_path)
    human_scanpaths   = loader.load_human_scanpaths(dataset_info['scanpaths_dir'], human_subject)
    config            = loader.load_config(constants.CONFIG_DIR, config_name, constants.IMAGE_SIZE, dataset_info['max_scanpath_length'], number_of_processes, save_probability_maps, human_scanpaths, checkpoint)
    trials_properties = loader.load_trials_properties(trials_properties_file, image_name, image_range, human_scanpaths, checkpoint)
    # breakpoint()
    visualsearch.run(config, dataset_info, trials_properties, human_scanpaths, output_path, constants.SIGMA)
    
    # Agrego solo para poder calcular las metricas luego
    return config['search_model']
    
""" Main method, added to be polymorphic with respect to the other models """
def main(dataset_name, config_name=constants.CONFIG_NAME, human_subject=None, metrics=None, models=None):
    models = setup_and_run(dataset_name, config_name=config_name, image_name=None, image_range=None, human_subject=human_subject, number_of_processes=constants.NUMBER_OF_PROCESSES, save_probability_maps=False)
    # breakpoint()
    if metrics:
        cum_perf   = 'perf' in metrics
        multimatch = 'mm' in metrics
        human_scanpath_prediction = 'hsp' in metrics
        # por ahora el elm y el nnIBS son lo mismo por como esta organizado el codigo
        if models == 'elm':
            models = ['nnIBS']
        # el main de metrics espera una lista de modelos no un string de un solo modelo
        if ~isinstance(dataset_name,list):
            dataset_name = [dataset_name]
        metrics_module.main(dataset_name, models, cum_perf, multimatch, human_scanpath_prediction)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    available_models   = utils.get_dirs(global_constants.MODELS_PATH)
    available_datasets = utils.get_dirs(global_constants.DATASETS_PATH)
    available_metrics  = global_constants.AVAILABLE_METRICS
    parser.add_argument('--d', '--datasets', type=str, nargs='*', default='Interiors', help='Names of the datasets on which to run the models. \
        Values must be in list: ' + str(available_datasets))
    parser.add_argument('--c', '--config', type=str, nargs='*', default='elm', help='Names of the models to run. \
        Values must be in list: ' + str(available_models))
    parser.add_argument('--m', '--metrics', type=str, nargs='*', default='perf mm', help='Names of the metrics to calculate. \
        Values must be in list: ' + str(available_metrics))
    args = parser.parse_args()
    
    main(args.d, config_name=args.c, metrics=args.m)