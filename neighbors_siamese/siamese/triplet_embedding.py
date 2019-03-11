import numpy as np
import os
import datetime
import multiprocessing
import keras
import tensorflow as tf
import keras.models as KM
import keras.layers as KL
from keras.optimizers import Adam
import re

import utils
from dataset import TripletDataset, TripletDataGenerator

def triplet_loss(input, margin=1.0):
    anchor = input[0]
    positive = input[1]
    negative = input[2]

    pos_dist = tf.square(anchor - positive)
    neg_dist = tf.square(anchor - negative)

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), margin)
    loss = tf.maximum(basic_loss, 0.0)

    return loss

def TripletEmbeddingNetwork(input_shape, layer_sizes, weights_filename=None):

    input = KL.Input(shape=input_shape, name='embedding_input')

    # Dense layer with ReLU activation for all but the last layer.
    layer_index = 1
    embedding = input
    for layer_size in layer_sizes[:-1]:
        embedding = KL.Dense(layer_size, activation='relu', name='fc{0}'.format(layer_index))(embedding)
        layer_index += 1

    # No activation for the last layer, since this represents the embedding.
    embedding = KL.Dense(layer_sizes[-1], name='output_embedding')(embedding)

    model = KM.Model(inputs=input, outputs=embedding, name='triplet_embedding_network')

    if weights_filename:
        model.load_weights(weights_filename, by_name=True)

    return model

class TripletEmbedding(object):

    def __init__(self, mode, config):
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        model_path, model_name = os.path.split(config['model_filename'])
        model_name, _ = os.path.splitext(model_name)
        self.set_log_dir(model_path, model_name)
        self.keras_model = self.build(mode=mode, config=config)
        print(self.keras_model.summary())

    def build(self, mode, config):
        """Build siamese network architecture"""
        input_shape = config['input_shape']

        # Set up inputs
        anchor_in = KL.Input(shape = input_shape, name = 'Anchor_Input')
        positive_in = KL.Input(shape = input_shape, name = 'Positive_Input')
        negative_in = KL.Input(shape = input_shape, name = 'Negative_Input')

        layers = []

        # Dense layer with ReLU activation for all but the last layer.
        layer_index = 1
        for layer_size in config['layer_sizes'][:-1]:
            layers.append(KL.Dense(layer_size, activation='relu', name='fc{0}'.format(layer_index)))
            layer_index += 1

        # No activation for the last layer, since this represents the embedding.
        layers.append(KL.Dense(config['layer_sizes'][-1], name='output_embedding'))

        embeddings = []

        # embedding_network = TripletEmbeddingNetwork(input_shape, config['layer_sizes'], config['embedding_weights'])
        #
        # embeddings.append(embedding_network(anchor_in))
        # embeddings.append(embedding_network(positive_in))
        # embeddings.append(embedding_network(negative_in))

        for input in [anchor_in, positive_in, negative_in]:
            embedding = input
            for layer in layers:
                embedding = layer(embedding)
            embeddings.append(embedding)

        output = KL.Lambda(triplet_loss, name='triplet_loss')(embeddings)

        keras_model = KM.Model(inputs=[anchor_in, positive_in, negative_in],
                            outputs=output,
                            name='triplet_embedding_network')

        # if mode == 'inference' or config['weights'] == 'last':
        #     keras_model.load_weights(config['model_filename'])

        return keras_model

    def predict(self, image_pairs, batch_size=32):
        return self.keras_model.predict(image_pairs, batch_size=batch_size)

    def train(self, config):
        train_dataset = TripletDataset(config['dataset_path'], 'train', config['data_augmentation_suffixes'])
        train_dataset.prepare(config['num_train_pairs'])

        val_dataset = TripletDataset(config['dataset_path'], 'validation')
        val_dataset.prepare(config['num_val_pairs'])

        train_generator = TripletDataGenerator(train_dataset, batch_size=config['batch_size'],
                                                       dim=self.config['input_shape'],
                                                       shuffle=config['shuffle_training_inputs'])
        val_generator = TripletDataGenerator(val_dataset, batch_size=config['batch_size'],
                                                   dim=self.config['input_shape'],
                                                   shuffle=config['shuffle_training_inputs'])

        model_path, _ = os.path.split(self.config['model_filename'])
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path, verbose=0, save_weights_only=True)
        ]

        self.keras_model.compile(loss=utils.l2_loss,
                                 optimizer=Adam(lr=config['learning_rate']))

        self.keras_model.fit_generator(generator=train_generator,
                                validation_data=val_generator,
                                epochs=config['epochs'],
                                use_multiprocessing=True,
                                callbacks=callbacks,
                                workers=multiprocessing.cpu_count())

        self.keras_model.save(self.config['model_filename'])


    def set_log_dir(self, model_path, model_name):
        """Sets the model log directory and epoch counter.
        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/[\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(model_path, "{}{:%Y%m%dT%H%M}".format(
            model_name, now))

        # Create log_dir if not exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            model_name.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")
