from os import path
# Configuration constants
# All paths are relative to root

CONFIG_NAME   = 'default'
CONFIG_DIR    = path.join('Models', 'ELM', 'configs')
DATASETS_PATH = 'Datasets'
RESULTS_PATH  = 'Results'
SALIENCY_PATH = path.join('Models', 'ELM', 'data', 'saliency')
TARGET_SIMILARITY_PATH = path.join('Models', 'ELM', 'data', 'target_similarity_maps')

NUMBER_OF_PROCESSES = 'all'
SIGMA      = [[4000, 0], [0, 2600]]
IMAGE_SIZE = (768, 1024)
