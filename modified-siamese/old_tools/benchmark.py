import numpy as np
import os
import datetime
import argparse
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm
import time

from autolab_core import YamlConfig

from siamese import SiameseNetwork
from siamese.utils import l2_distance, l1_distance, accuracy, compute_accuracy, set_tf_config, show_output
from siamese.dataset import ImageDataset, DataGenerator

def benchmark(config):
    model = SiameseNetwork('inference', config['model'])

    params = {
        'batch_size' : config['benchmark']['batch_size'],
        'shuffle': False,
        'dim': config['model']['input_shape']
    }
    dataset_path = config['benchmark']['dataset_path']
    train_dataset = ImageDataset(dataset_path, 'train')
    train_dataset.prepare(config['benchmark']['test_cases']//2)
    train_generator = DataGenerator(train_dataset, **params)
    test_dataset = ImageDataset(dataset_path, 'validation')
    test_dataset.prepare(config['benchmark']['test_cases']//2)
    test_generator = DataGenerator(test_dataset, **params)

    preds = np.array([])
    gts = np.array([])
    for i in tqdm(range(len(train_generator))):
        batch = train_generator[i]
        pred = model.predict(batch[0])
        preds = np.append(preds, pred.flatten())
        gts = np.append(gts, batch[1])
        # if config['vis_output'] and not i % config['test_cases']//(5*config['batch_size']):
        #     show_output(batch[0][0], batch[0][1], pred, batch[1])
    tr_acc = compute_accuracy(preds, gts)
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))

    evaluation_times = []
    preds = np.array([])
    gts = np.array([])
    for i in tqdm(range(len(test_generator))):
        batch = test_generator[i]
        start_t = time.time()
        pred = model.predict(batch[0])

        evaluation_times.append(   (time.time() - start_t) /len(batch)   )

        preds = np.append(preds, pred.flatten())
        gts = np.append(gts, batch[1])
        if config['benchmark']['vis_output'] and not i % config['benchmark']['test_cases']//(5*config['benchmark']['batch_size']):
            show_output(batch[0][0], batch[0][1], pred, batch[1])
    te_acc = compute_accuracy(preds, gts)

    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    print ("Average Evaluation Time Per Sample: " + str(  np.mean(evaluation_times)  ))

    print(preds)
    print(gts)

if __name__ == "__main__":

    # parse the provided configuration file, set tf settings, and train
    conf_parser = argparse.ArgumentParser(description="Benchmark Siamese model")
    conf_parser.add_argument("--config", action="store", default="cfg/benchmark.yaml",
                               dest="conf_file", type=str, help="path to the configuration file")
    conf_args = conf_parser.parse_args()

    # read in config file information from proper section
    config = YamlConfig(conf_args.conf_file)
    # set_tf_config()
    benchmark(config)
