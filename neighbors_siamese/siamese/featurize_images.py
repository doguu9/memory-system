"""
Utility script to save ResNet50 featurizations of images. Given a dataset in
/path/to/dataset, this script will create a dataset in /path/to/dataset_features
containing the same structure as the original dataset, but instead contain .npz
files containing the featurizations.
"""
import argparse
import numpy as np
import os
import multiprocessing
import skimage.io

import utils
from resnet_fused import ResNet50Fused

# Limit to one GPU.
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Limit number of CPU cores used.
cpu_cores = [8, 9, 10, 11] # Cores (numbered 0-11)
os.system("taskset -pc {} {}".format(",".join(str(i) for i in cpu_cores), os.getpid()))

def featurize_dataset_dir(dataset_dir, output_dir, resnet_model):
    print("Processing Directory:", dataset_dir)

    dir_contents = os.listdir(dataset_dir)

    # Recursively featurized subdirectories.
    if os.path.isdir(os.path.join(dataset_dir, dir_contents[0])):
        for subdir in dir_contents:
            dataset_subdir = os.path.join(dataset_dir, subdir)
            output_subdir = os.path.join(output_dir, subdir)
            os.makedirs(output_subdir)
            featurize_dataset_dir(dataset_subdir, output_subdir, resnet_model)
    # If images are in this subdirectory, directly featurize them and save them in
    # compressed .npz format.
    else:
        for img_path in dir_contents:
            full_image_path = os.path.join(dataset_dir, img_path)
            image = skimage.io.imread(full_image_path)
            features = np.squeeze(resnet_model.predict(image[None,:]))
            img_path_base, _ = os.path.splitext(img_path)
            full_output_path = os.path.join(output_dir, "{0}.npz".format(img_path_base))
            np.savez_compressed(full_output_path, features)


def featurize_dataset(dataset_dir):
    feature_dir = "{0}_{1}".format(dataset_dir, 'features')

    if os.path.exists(feature_dir):
        raise Exception("Featurization directory already exists!")

    os.makedirs(feature_dir)

    resnet_model = ResNet50Fused(include_top=False, weights='imagenet', input_shape= (512, 512, 3)) #(772, 1032, 3))
    featurize_dataset_dir(dataset_dir, feature_dir, resnet_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='Path to dataset to featurize.')
    args = parser.parse_args()

    featurize_dataset(args.dataset)
