import numpy as np
import os
import datetime
import argparse
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm
import time

import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Input,Dense, Lambda

from autolab_core import YamlConfig

from utils import l2_distance, l1_distance, accuracy, compute_accuracy, set_tf_config, show_output
from dataset import ImageDataset, DataGenerator
from resnet_fused import ResNet50Fused
import pickle

def benchmark(config):

    model_config = config['model']
    benchmark_config = config['benchmark']
    model_path = model_config['model_filename']
    dataset_path = benchmark_config['dataset_path']

    # Parameters
    params = {'dim': model_config['input_shape'],
            'batch_size': benchmark_config['batch_size'],
            'shuffle': False}

    # Dataset
    train_dataset = ImageDataset(dataset_path, 'train')
    train_dataset.prepare(benchmark_config['test_cases']//2)
    train_generator = DataGenerator(train_dataset, **params)
    test_dataset = ImageDataset(dataset_path, 'validation')
    test_dataset.prepare(benchmark_config['test_cases']//2)
    test_generator = DataGenerator(test_dataset, **params)

    if os.path.exists(model_path):
        feature_model = ResNet50Fused(include_top=False, weights=None, input_shape=model_config['input_shape'])
        img_a_in = Input(shape = model_config['input_shape'], name = 'ImageA_Input')
        img_b_in = Input(shape = model_config['input_shape'], name = 'ImageB_Input')
        if model_config['two_gpus']:
            with tf.device('/gpu:0'):
                img_a_feat = feature_model(img_a_in)
            with tf.device('/gpu:1'):
                img_b_feat = feature_model(img_b_in)
        else:
            img_a_feat = feature_model(img_a_in)
            img_b_feat = feature_model(img_b_in)
        distance = Lambda(l1_distance)([img_a_feat, img_b_feat])
        distance = Dense(1, activation = 'sigmoid')(distance)
        similarity_model = Model(inputs = [img_a_in, img_b_in], outputs = distance, name = 'Similarity_Model')
        similarity_model.load_weights(model_path)
    else:
        print('No model exists at the given path!')
        exit(1)

    # preds = np.array([])
    # gts = np.array([])
    # for i in tqdm(range(len(train_generator))):
    #     batch = train_generator[i]
    #     pred = similarity_model.predict_on_batch(batch[0])
    #     preds = np.append(preds, pred.flatten())
    #     gts = np.append(gts, batch[1])
    #     # if config['vis_output'] and not i % config['test_cases']//(5*config['batch_size']):
    #     #     show_output(batch[0][0], batch[0][1], pred, batch[1])
    # tr_acc = compute_accuracy(preds, gts)
    # print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))

    evaluation_times = []
    preds = np.array([])
    gts = np.array([])
    for i in tqdm(range(len(test_generator))):
        batch = test_generator[i]
        start_t = time.time()
        pred = similarity_model.predict_on_batch(batch[0])

        evaluation_times.append(   (time.time() - start_t) /len(batch)   )

        preds = np.append(preds, pred.flatten())
        gts = np.append(gts, batch[1])
        if benchmark_config['vis_output'] and not i % benchmark_config['test_cases']//(5*benchmark_config['batch_size']):
            show_output(batch[0][0], batch[0][1], pred, batch[1])
    te_acc = compute_accuracy(preds, gts)

    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    print ("Average Evaluation Time Per Sample: " + str(  np.mean(evaluation_times)  ))

    # print(preds)
    # print(gts)

def run_get_object_confusion(config):

    model_config = config['model']
    model_path = model_config['model_filename']
    benchmark_config = config['benchmark']
    dataset_path = benchmark_config['dataset_path']

    if os.path.exists(model_path):
        feature_model = ResNet50Fused(include_top=False, weights=None, input_shape=model_config['input_shape'])
        img_a_in = Input(shape = model_config['input_shape'], name = 'ImageA_Input')
        img_b_in = Input(shape = model_config['input_shape'], name = 'ImageB_Input')
        if model_config['two_gpus']:
            with tf.device('/gpu:0'):
                img_a_feat = feature_model(img_a_in)
            with tf.device('/gpu:1'):
                img_b_feat = feature_model(img_b_in)
        else:
            img_a_feat = feature_model(img_a_in)
            img_b_feat = feature_model(img_b_in)
        distance = Lambda(l1_distance)([img_a_feat, img_b_feat])
        distance = Dense(1, activation = 'sigmoid')(distance)
        similarity_model = Model(inputs = [img_a_in, img_b_in], outputs = distance, name = 'Similarity_Model')
        similarity_model.load_weights(model_path)
    else:
        print('No model exists at the given path!')
        exit(1)

    orig_data_path = os.path.join(dataset_path, 'originals')
    class_labels = os.listdir(orig_data_path)

    distinguishability_metrics = {}

    for class1 in class_labels:
    	distinguishability_metric = 0
    	for class2 in class_labels:
    		if class1 != class2:
    			distinguishability_metric += get_object_confusion(class1, class2, similarity_model, config)

    	distinguishability_metrics[class1] = distinguishability_metric
    	print("Object: " + class1 + ", Distinguishability Metric: " + str(distinguishability_metric) )

    pickle.dump( distinguishability_metrics, open( "distinguishability_metrics.p", "wb" ) )

def get_object_confusion(class1, class2, similarity_model, config):

    model_config = config['model']
    benchmark_config = config['benchmark']
    model_path = model_config['model_filename']
    dataset_path = benchmark_config['dataset_path']

    params = {'dim': model_config['input_shape'],
            'batch_size': benchmark_config['batch_size'],
            'shuffle': False}

    test_dataset = ImageDataset(dataset_path, 'validation')
    test_dataset.prepare_specific(benchmark_config['test_cases']//2, class1, class2)
    test_generator = DataGenerator(test_dataset, **params)
    preds = np.array([])
    gts = np.array([])

    for i in tqdm(range(len(test_generator))):
        batch = test_generator[i]
        pred = similarity_model.predict_on_batch(batch[0])
        preds = np.append(preds, pred.flatten())
        gts = np.append(gts, batch[1])
        if benchmark_config['vis_output'] and not i % benchmark_config['test_cases']//(5*benchmark_config['batch_size']):
            show_output(batch[0][0], batch[0][1], pred, batch[1])
    te_acc = compute_accuracy(preds, gts)
    print("Class 1: " + class1 + ", Class2: " + class2 + ", Distinguishability Score: " + str(te_acc) )

    return te_acc

if __name__ == "__main__":

    # parse the provided configuration file, set tf settings, and train
    conf_parser = argparse.ArgumentParser(description="Benchmark Siamese model")
    conf_parser.add_argument("--config", action="store", default="../cfg/benchmark.yaml",
                               dest="conf_file", type=str, help="path to the configuration file")
    conf_args = conf_parser.parse_args()

    # read in config file information from proper section
    config = YamlConfig(conf_args.conf_file)
    # set_tf_config()

    # benchmark(config)
    run_get_object_confusion(config)
