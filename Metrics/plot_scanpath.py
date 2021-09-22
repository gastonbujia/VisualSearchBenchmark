import constants
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from os import makedirs, path, listdir
from random import randint
from skimage import io, transform
from scripts import utils
import argparse
import sys

""" Usage:
    To plot a model's scanpath on a given image:
        plot_scanpath.py -dataset <dataset_name> -img <image_name> -model <model_name>
    To plot a (random) human subject's scanpath on a given image:
        plot_scanpath.py -dataset <dataset_name> -img <image_name> -human
"""

""" The main method of this script (plot_scanpath) belongs to https://github.com/cvlab-stonybrook/Scanpath_Prediction/plot_scanpath.py """

def plot_scanpath(img, xs, ys, fixation_size, bbox, title, save_path):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=plt.cm.gray)

    for i in range(len(xs)):
        if i > 0:
            plt.arrow(xs[i - 1], ys[i - 1], xs[i] - xs[i - 1], ys[i] - ys[i - 1], width=3, color='yellow', alpha=0.5)

    for i in range(len(xs)):
        circle = plt.Circle((xs[i], ys[i]),
                            radius=fixation_size[1] // 2,
                            edgecolor='red',
                            facecolor='yellow',
                            alpha=0.5)
        ax.add_patch(circle)
        plt.annotate("{}".format(i + 1), xy=(xs[i], ys[i] + 3), fontsize=10, ha="center", va="center")

    # Draw target's bbox
    rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], alpha=0.7, edgecolor='red', facecolor='none', linewidth=2)
    ax.add_patch(rect)

    # To draw grid, useful for plotting cIBS's scanpaths
    # box_size = 32
    # box_x = 0
    # box_y = 0
    # rows = round(img.shape[0] / box_size)
    # columns = round(img.shape[1] / box_size)
    # for row in range(rows):
    #     box_y = box_size * row
    #     for column in range(columns):
    #         box_x = box_size * column
    #         rect = Rectangle((box_x, box_y), box_size, box_size, alpha=0.5, edgecolor='yellow', facecolor='none', linewidth=2)
    #         ax.add_patch(rect)


    ax.axis('off')
    ax.set_title(title)

    plt.savefig(path.join(save_path, title + '.png'))
    plt.show()
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group()
    group.add_argument('-model', type=str, help='Name of the visual search model')
    group.add_argument('-human', action='store_true', help='Flag which indicates to plot a scanpath generated by a random human subject (who has found the target)')
    parser.add_argument('-dataset', type=str, help='Name of the dataset')
    parser.add_argument('-img', type=str, help='Name of the image on which to draw the scanpath (write \'notfound\' to plot target not found images')

    args = parser.parse_args()
    return args

def get_trial_info(image_name, trials_properties):
    for trial in trials_properties:
        if trial['image'] == image_name:
            return trial        

    raise NameError('Image name must be in the dataset')

def rescale_coordinate(value, old_size, new_size, fixation_size=None, is_grid=False):
    if is_grid:
        # Rescale fixation to center of the cell in the grid
        return value * fixation_size + (fixation_size // 2)
    else:
        return (value / old_size) * new_size

def process_image(img_scanpath, subject, image_name, dataset_name, trial_info, images_path):
    fixation_size     = (img_scanpath['receptive_height'], img_scanpath['receptive_width'])
    scanpath_img_size = (img_scanpath['image_height'], img_scanpath['image_width'])

    image_file = path.join(images_path, image_name)
    img        = io.imread(image_file)
    img_size_used = scanpath_img_size
    original_img_size = img.shape[:2]

    is_grid = False
    # cIBS uses a grid for images, it's necessary to upscale it
    if subject == 'cIBS':
        is_grid = True
        img_size_used = (768, 1024)
        fixation_size = (img_size_used[0] // scanpath_img_size[0], img_size_used[1] // scanpath_img_size[1])

    img = transform.resize(img, img_size_used)
    # Rescale scanpath if necessary
    X = [rescale_coordinate(x, scanpath_img_size[1], img_size_used[1], fixation_size[1], is_grid) for x in img_scanpath['X']]
    Y = [rescale_coordinate(y, scanpath_img_size[0], img_size_used[0], fixation_size[0], is_grid) for y in img_scanpath['Y']]

    #bbox = [trial_info['target_matched_row'], trial_info['target_matched_column'], trial_info['target_matched_row'] + trial_info['target_height'], \
    #    trial_info['target_matched_column'] + trial_info['target_width']]
    bbox = img_scanpath['target_bbox']
    
    bbox[0], bbox[2] = [rescale_coordinate(pos, original_img_size[0], scanpath_img_size[0], fixation_size[0], is_grid) for pos in (bbox[0], bbox[2])]
    bbox[1], bbox[3] = [rescale_coordinate(pos, original_img_size[1], scanpath_img_size[1], fixation_size[1], is_grid) for pos in (bbox[1], bbox[3])]
    target_height = bbox[2] - bbox[0]
    target_width  = bbox[3] - bbox[1]
    bbox = [bbox[1], bbox[0], target_width, target_height]

    save_path = path.join('Plots', path.join(dataset_name + '_dataset', image_name[:-4]))
    if not path.exists(save_path):
        makedirs(save_path)

    title = image_name[:-4] + '_' + subject.replace(' ', '_')

    plot_scanpath(img, X, Y, fixation_size, bbox, title, save_path)

if __name__ == '__main__':
    args = parse_args()

    if not args.human:
        scanpaths_dir = path.join(path.join(constants.RESULTS_DIR, args.dataset + '_dataset'), args.model)
        if not path.exists(scanpaths_dir):
            print('There are no results for ' + args.model + ' in the ' + args.dataset + ' dataset')
            sys.exit(0)

        scanpaths_file = path.join(scanpaths_dir, 'Scanpaths.json')
        scanpaths      = utils.load_dict_from_json(scanpaths_file)
        
        if args.img != 'notfound':
            if not args.img in scanpaths:
                print('Image not found in ' + args.model + ' scanpaths')
                sys.exit(0)
            img_scanpath = scanpaths[args.img]
        subject = args.model
    else:
        human_scanpaths_dir = path.join(path.join(constants.DATASETS_DIR, args.dataset), 'human_scanpaths')
        if not path.exists(human_scanpaths_dir) or not listdir(human_scanpaths_dir):
            print('There are no human subjects scanpaths for this dataset')
            sys.exit(0)
        
        human_scanpaths_files = listdir(human_scanpaths_dir)
        number_of_subjects    = len(human_scanpaths_files)
        human_subject         = randint(0, number_of_subjects - 1)
        human_scanpaths_files.sort()

        target_found = False
        checked_subjects = []
        while not target_found:
            scanpaths_file = path.join(human_scanpaths_dir, human_scanpaths_files[human_subject])
            scanpaths      = utils.load_dict_from_json(scanpaths_file)
            
            if args.img in scanpaths:
                img_scanpath = scanpaths[args.img]
                target_found = img_scanpath['target_found']
            if not target_found:
                checked_subjects.append(human_subject)
                if len(checked_subjects) == number_of_subjects:
                    print('No human subject has had a successful trial for image ' + args.img)
                    sys.exit(0)

                human_subject = randint(0, number_of_subjects - 1)
                while human_subject in checked_subjects:
                    human_subject = randint(0, number_of_subjects - 1)
        
        subject = 'Human subject ' + str(human_subject + 1)
    
    dataset_path = path.join(constants.DATASETS_DIR, args.dataset)
    dataset_info = utils.load_dict_from_json(path.join(dataset_path, 'dataset_info.json'))
    
    images_path = path.join(dataset_path, dataset_info['images_dir'])

    trials_properties_file = path.join(dataset_path, 'trials_properties.json')
    trials_properties      = utils.load_dict_from_json(trials_properties_file)
    
    trial_info = get_trial_info(args.img, trials_properties)

    if args.img == 'notfound' and not args.human:
        for image_name in scanpaths.keys():
            if not scanpaths[image_name]['target_found']:
                process_image(scanpaths[image_name], subject, image_name, args.dataset, trial_info, images_path)
    else:
        process_image(img_scanpath, subject, args.img, args.dataset, trial_info, images_path)