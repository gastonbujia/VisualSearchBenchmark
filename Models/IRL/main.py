""" Script which runs the IRL model on a given dataset """

import torch
import numpy as np
from . import constants
from tqdm import tqdm
from os import path, cpu_count
from .dataset import process_eval_data, process_trials, load_human_scanpaths
from .irl_dcb.config import JsonConfig
from torch.utils.data import DataLoader
from .irl_dcb.models import LHF_Policy_Cond_Small
from .irl_dcb.environment import IRL_Env4LHF
from .irl_dcb import utils

torch.manual_seed(42619)
np.random.seed(42619)

def main(dataset_name, human_subject=None):
    device  = torch.device('cpu')
    hparams = path.join(constants.HPARAMS_PATH, 'default.json')
    hparams = JsonConfig(hparams)

    dcbs_path = path.join(constants.DCBS_PATH, dataset_name)
    # Dir of high and low res belief maps
    DCB_dir_HR = path.join(dcbs_path, 'DCBs', 'HR')
    DCB_dir_LR = path.join(dcbs_path, 'DCBs', 'LR')
    
    dataset_path = path.join(constants.DATASETS_PATH, dataset_name)

    dataset_info      = utils.load_dict_from_json(path.join(dataset_path, 'dataset_info.json'))
    trials_properties = utils.load_dict_from_json(path.join(dataset_path, 'trials_properties.json'))
    output_path       = path.join(constants.RESULTS_PATH, dataset_name + '_dataset', 'IRL')
    
    images_dir     = path.join(dataset_path, dataset_info['images_dir'])
    new_image_size = (hparams.Data.im_h, hparams.Data.im_w)
    grid_size      = (hparams.Data.patch_num[1], hparams.Data.patch_num[0])

    # For computing different metrics; used only through argument --h
    human_scanpaths_dir = path.join(dataset_path, dataset_info['scanpaths_dir'])
    human_scanpaths     = load_human_scanpaths(human_scanpaths_dir, human_subject, grid_size)
    if human_scanpaths:
        human_subject_str = '0' + str(human_subject) if human_subject < 10 else str(human_subject)
        output_path = path.join(output_path, 'subjects_predictions', 'subject_' + human_subject_str)

    hparams.Data.max_traj_length = dataset_info['max_scanpath_length'] - 1

    # Process trials, creating belief maps when necessary, and get target's bounding box for each trial
    bbox_annos = process_trials(trials_properties, images_dir, human_scanpaths, new_image_size, grid_size, DCB_dir_HR, DCB_dir_LR)

    # Get categories and load image data
    dataset = process_eval_data(trials_properties, human_scanpaths, DCB_dir_HR, DCB_dir_LR, bbox_annos, grid_size, hparams)
    
    batch_size = min(len(trials_properties), 64)
    img_loader = DataLoader(dataset['img_test'],
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=cpu_count())

    print('Number of images: ', len(dataset['img_test']))

    # Load trained model
    task_eye  = torch.eye(len(dataset['catIds'])).to(device)
    generator = LHF_Policy_Cond_Small(hparams.Data.patch_count,
                                      len(dataset['catIds']), task_eye,
                                      constants.NUMBER_OF_BELIEF_MAPS).to(device)
    
    utils.load('best', generator, 'generator', pkg_dir=constants.TRAINED_MODELS_PATH, device=device)
    generator.eval()

    # Build environment
    env_test = IRL_Env4LHF(hparams.Data,
                           max_step=hparams.Data.max_traj_length,
                           mask_size=hparams.Data.IOR_size,
                           status_update_mtd=hparams.Train.stop_criteria,
                           device=device,
                           inhibit_return=True,
                           init_mtd='manual')

    # Generate scanpaths
    print('Generating scanpaths...')
    all_actions = gen_scanpaths(generator,
                                env_test,
                                img_loader,
                                bbox_annos,
                                hparams.Data.patch_num,
                                hparams.Data.patch_size,
                                hparams.Data.max_traj_length,
                                hparams.Data.im_w,
                                hparams.Data.im_h,
                                human_scanpaths,
                                num_sample=1,
                                output_path=output_path)
    
    scanpaths     = utils.actions2scanpaths(all_actions, hparams.Data.patch_num, hparams.Data.patch_size, hparams.Data.im_w, hparams.Data.im_h, dataset_name, hparams.Data.max_traj_length)
    targets_found = utils.cutFixOnTarget(scanpaths, bbox_annos, hparams.Data.patch_size)

    print('Total targets found: ' + str(targets_found) + '/' + str(len(scanpaths)))

    if human_scanpaths:
        utils.save_scanpaths(output_path, human_scanpaths, filename='Subject_scanpaths.json')
    else:    
        utils.save_scanpaths(output_path, scanpaths)

def gen_scanpaths(generator, env_test, test_img_loader, bbox_annos, patch_num, patch_size, max_traj_len, im_w, im_h, human_scanpaths, num_sample, output_path):
    all_actions = []
    for i_sample in range(num_sample):
        progress = tqdm(test_img_loader)
        for i_batch, batch in enumerate(progress):
            img_names_batch   = batch['img_name']
            cat_names_batch   = batch['cat_name']
            initial_fix_batch = batch['init_fix']
            human_scanpaths_batch = []
            max_traj_len_batch = max_traj_len
            if human_scanpaths:
                human_scanpaths_batch = [human_scanpaths[image_name] for image_name in img_names_batch]
                max_traj_len_batch    = utils.get_max_scanpath_length(human_scanpaths_batch) - 1
                # Check for precomputed probability maps
                if utils.probability_maps_for_batch(img_names_batch, output_path):
                    utils.save_and_compute_metrics(None, human_scanpaths_batch, img_names_batch, output_path, presaved=True)
                    continue

            env_test.set_data(batch, max_traj_len_batch)
            with torch.no_grad():
                env_test.reset(max_traj_len_batch)
                trajs, probs = utils.collect_trajs(env_test,
                                            generator,
                                            img_names_batch,
                                            patch_num,
                                            max_traj_len_batch,
                                            human_scanpaths_batch,
                                            is_eval=True,
                                            sample_action=True)
                
                if human_scanpaths:
                    utils.save_and_compute_metrics(probs, human_scanpaths_batch, img_names_batch, output_path)
                all_actions.extend([(cat_names_batch[i], img_names_batch[i], initial_fix_batch[i],
                                     'present', trajs['actions'][:, i])
                                    for i in range(env_test.batch_size)])
    
    return all_actions