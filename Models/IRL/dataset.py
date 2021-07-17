import numpy as np
import pandas as pd
from irl_dcb.data import LHF_IRL

def process_eval_data(trials_properties,
                 DCB_HR_dir,
                 DCB_LR_dir,
                 target_annos,
                 hparams):
    target_init_fixs = {}
    for image_data in trials_properties:
        key = image_data['target_object'] + '_' + image_data['image']
        target_init_fixs[key] = (image_data['initial_fixation_column'] / image_data['image_width'],
                                image_data['initial_fixation_row'] / image_data['image_height'])

    # categories_list = pd.read_csv('Lista categorías.csv')
    # things = categories_list['things'].to_numpy(dtype=str)
    # stuff  = categories_list['stuff'].to_numpy(dtype=str)[:-26]
    # things_stuff = np.concatenate((things, stuff))

    target_objects = list(np.unique([x['target_object'] for x in trials_properties]))
    # target_objects = things_stuff
    catIds = dict(zip(target_objects, list(range(len(target_objects)))))

    test_task_img_pair = np.unique(
            [traj['target_object'] + '_' + traj['image'] for traj in trials_properties])

    # Load image data
    test_img_dataset = LHF_IRL(DCB_HR_dir, DCB_LR_dir, target_init_fixs,
                                   test_task_img_pair, target_annos,
                                   hparams.Data, catIds)
    return {
            'catIds': catIds,
            'img_test': test_img_dataset,
        }