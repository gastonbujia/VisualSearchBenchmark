import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from os import makedirs, path, listdir
from random import randint
from skimage import io, transform
import argparse
import json
import sys

""" The main method of this script (plot_scanpath) belongs to https://github.com/cvlab-stonybrook/Scanpath_Prediction/plot_scanpath.py """

datasets_dir = '../Datasets/'
results_dir  = '../Results/'

def plot_scanpath(img, xs, ys, fixation_size, bbox=None, title=None, save_path=None):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=plt.cm.gray)

    for i in range(len(xs)):
        if i > 0:
            plt.arrow(xs[i - 1], ys[i - 1], xs[i] - xs[i - 1], ys[i] - ys[i - 1], width=3, color='yellow', alpha=0.5)

    for i in range(len(xs)):
        circle = plt.Circle((xs[i], ys[i]),
                            radius=fixation_size // 2,
                            edgecolor='red',
                            facecolor='yellow',
                            alpha=0.5)
        ax.add_patch(circle)
        plt.annotate("{}".format(i + 1), xy=(xs[i], ys[i] + 3), fontsize=10, ha="center", va="center")

    if bbox is not None:
        rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], alpha=0.7, edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(rect)

    # Para graficar grilla, en el caso de cIBS
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
    if title is not None:
        ax.set_title(title)
    if save_path is not None:
        plt.savefig(path.join(save_path, title + '.png'))
    plt.show()
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-model', type=str, help='Name of the visual search model')
    group.add_argument('-human', action='store_true', help='Flag which indicates to plot a scanpath generated by a random human subject (who has found the target)')
    parser.add_argument('-dataset', type=str, help='Name of the dataset')
    parser.add_argument('-img', type=str, help='Name of the image on which to draw the scanpath (write \'notfound\' to plot target not found images')

    args = parser.parse_args()
    return args

def process_image(img_scanpath, subject, image_path, dataset_name):
    X = img_scanpath['X']
    Y = img_scanpath['Y']

    bbox = img_scanpath['target_bbox']
    target_height = bbox[2] - bbox[0]
    target_width  = bbox[3] - bbox[1]
    bbox = [bbox[1], bbox[0], target_width, target_height]

    fixation_size = img_scanpath['receptive_width']
    
    # TODO: Levantar del JSON del dataset
    if dataset_name == 'IVSN':
        image_folder = 'stimuli'
    else:
        image_folder = 'images'

    image_size = (img_scanpath['image_height'], img_scanpath['image_width'])
    image_file = datasets_dir + dataset_name + '/' + image_folder + '/' + image_path
    img = io.imread(image_file)
    img = transform.resize(img, image_size)

    save_path = path.join('Plots', dataset_name)
    if not path.exists(save_path):
        makedirs(save_path)

    title = subject.replace(' ', '_') + '_' + image_path[:-4]

    plot_scanpath(img, X, Y, fixation_size, bbox, title, save_path)

if __name__ == '__main__':
    args = parse_args()

    if not args.human:
        scanpaths_dir = path.join(results_dir + args.dataset + '_dataset', args.model)
        if not path.exists(scanpaths_dir):
            print('There are no results for ' + args.model + ' in the ' + args.dataset + ' dataset')
            sys.exit(0)
        scanpaths_file = path.join(scanpaths_dir, 'Scanpaths.json')

        with open(scanpaths_file, 'r') as fp:
            scanpaths = json.load(fp)
        
        if args.img != 'notfound':
            if not args.img in scanpaths:
                print('Image not found in ' + args.model + ' scanpaths')
                sys.exit(0)
            img_scanpath = scanpaths[args.img]
        subject = args.model
    else:
        human_scanpaths_dir = path.join(datasets_dir + args.dataset, 'human_scanpaths')
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
            with open(scanpaths_file, 'r') as fp:
                scanpaths = json.load(fp)
            
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

    if args.img == 'notfound' and not args.human:
        for image_name in scanpaths.keys():
            if not scanpaths[image_name]['target_found']:
                process_image(scanpaths[image_name], subject, image_name, args.dataset)
    else:
        process_image(img_scanpath, subject, args.img, args.dataset)