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
from keras import models
from keras import layers
from keras import optimizers
import utils
from resnet_fused import ResNet50Fused

# Limit to one GPU.
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Limit number of CPU cores used.
cpu_cores = [8, 9, 10, 11] # Cores (numbered 0-11)
os.system("taskset -pc {} {}".format(",".join(str(i) for i in cpu_cores), os.getpid()))

train_dir = '/home/kate/proj1/imgs/phoxi_nop/train'
validation_dir = '/home/kate/proj1/imgs/phoxi_nop/validation'

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
            #image = np.load(full_image_path)
            #image = np.expand_dims(image, axis=2)
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
    for layer in resnet_model.layers[:-4]:
        layer.trainable = False
    for layer in resnet_model.layers:
        print(layer, layer.trainable)
    model = models.Sequential()
    model.add(resnet_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, acitvation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(3, activation='softmax'))
    model.summary()
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    train_batchsize = 50
    val_batchsize = 10
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='categorical')
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)
    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    # Train the model
    history = model.fit_generator(
          train_generator,
          steps_per_epoch=train_generator.samples/train_generator.batch_size ,
          epochs=30,
          validation_data=validation_generator,
          validation_steps=validation_generator.samples/validation_generator.batch_size,
          verbose=1)
    featurize_dataset_dir(dataset_dir, feature_dir, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='Path to dataset to featurize.')
    args = parser.parse_args()

    featurize_dataset(args.dataset)
