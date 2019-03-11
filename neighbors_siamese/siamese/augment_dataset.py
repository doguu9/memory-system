"""
Script to augment a dataset. The dataset folder should contain originals and train subdirectories.
This script will generate an augmented image for each image in the training set according to the
specified data augmentation options. These additional images will be created in subdirectories
originals_x and train_x in the same folder as the dataset and can be specified to be included during
training.
"""
import argparse
import data_augmentation as da
import numpy as np
import os
import scipy.misc
import skimage.io

# Limit number of CPU cores used.
cpu_cores = [0,1,2,3] # Cores (numbered 0-11)
os.system("taskset -pc {} {}".format(",".join(str(i) for i in cpu_cores), os.getpid()))

ORIGINALS_DIR = 'originals'
TRAIN_DIR = 'train'

def augment_image(image, v_flip, h_flip, rotate, scale):
    if v_flip:
        image = da.vertical_flip(image)

    if h_flip:
        image = da.horizontal_flip(image)

    # Rotate by a random angle between 0 and 360 degrees.
    if rotate:
        angle = np.random.uniform(0.0, 360.0)
        image = da.rotate(image, angle)

    # Randomly scale image down or up, each with 50% probability.
    if scale:
        if np.random.uniform() < 0.5:
            zoom_factor = np.random.uniform(0.5, 1.0)

        else:
            zoom_factor = np.random.uniform(1.0, 2.0)
        image = da.scale(image, zoom_factor)

    return image


def augment_dataset(args):
    originals_dir = os.path.join(args.dataset, ORIGINALS_DIR)
    train_dir = os.path.join(args.dataset, TRAIN_DIR)
    training_objects = os.listdir(train_dir)
    print("Training Objects:", training_objects)

    augmented_originals_dir = os.path.join(args.dataset, "{0}_{1}".format(ORIGINALS_DIR, args.output_suffix))
    augmented_train_dir = os.path.join(args.dataset, "{0}_{1}".format(TRAIN_DIR, args.output_suffix))

    if os.path.exists(augmented_originals_dir) or os.path.exists(augmented_train_dir):
        raise Exception("Data augmentation output directory already exists!")

    os.makedirs(augmented_originals_dir)
    os.makedirs(augmented_train_dir)

    for training_object in training_objects:
        print("Augmenting dataset for object:", training_object)
        obj_originals_dir = os.path.join(originals_dir, training_object)
        obj_original_files = os.listdir(obj_originals_dir)

        obj_originals_augmented_dir = os.path.join(augmented_originals_dir, training_object)
        os.makedirs(obj_originals_augmented_dir)

        for obj_original_file in obj_original_files:
            full_obj_original_path = os.path.join(obj_originals_dir, obj_original_file)
            image = skimage.io.imread(full_obj_original_path)
            augmented_image = augment_image(image, args.v_flip, args.h_flip, args.rotate, args.scale)

            output_path = os.path.join(obj_originals_augmented_dir, obj_original_file)
            scipy.misc.imsave(output_path, augmented_image)

        obj_train_dir = os.path.join(train_dir, training_object)
        augmented_train_object_dir = os.path.join(augmented_train_dir, training_object)
        os.makedirs(augmented_train_object_dir)
        views = os.listdir(obj_train_dir)

        for view in views:
            view_dir = os.path.join(obj_train_dir, view)
            augmented_view_dir = os.path.join(augmented_train_object_dir, view)
            os.makedirs(augmented_view_dir)

            image_files = os.listdir(view_dir)

            for file in image_files:
                full_original_path = os.path.join(view_dir, file)
                image = skimage.io.imread(full_original_path)
                augmented_image = augment_image(image, args.v_flip, args.h_flip, args.rotate, args.scale)

                output_path = os.path.join(augmented_view_dir, file)
                scipy.misc.imsave(output_path, augmented_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='Path to dataset (should contain originals and train subdirs.')
    parser.add_argument('output_suffix', type=str, help='This script will create originals_x and train_x subdirs in the dataset folder containing augmented images.')
    parser.add_argument('--v_flip', action='store_true', help='Perform a vertical flip.')
    parser.add_argument('--h_flip', action='store_true', help='Perform a horizontal flip.')
    parser.add_argument('--rotate', action='store_true', help='Perform a rotation (random angle between 0 and 360).')
    parser.add_argument('--scale', action='store_true', help='Scale the image (randomly from 1/2 of original size to 2 times the original size).')
    args = parser.parse_args()

    augment_dataset(args)
