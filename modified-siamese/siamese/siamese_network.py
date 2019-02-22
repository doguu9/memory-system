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
import matplotlib.pyplot as plt
import utils
from dataset import ImageDataset, DataGenerator
from resnet_fused import ResNet50Fused
from triplet_embedding import TripletEmbeddingNetwork

class SiameseNetwork(object):

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
        img_a_in = KL.Input(shape = input_shape, name = 'ImageA_Input')
        img_b_in = KL.Input(shape = input_shape, name = 'ImageB_Input')

        if config['input_type'] == 'images':
            # Set up featurization
            feature_model = None
            if mode == 'inference' or config['weights'] == 'random':
                feature_model = ResNet50Fused(include_top=False, weights=None, input_shape=input_shape)
            elif config['weights'] == 'imagenet':
                feature_model = ResNet50Fused(include_top=False, weights='imagenet', input_shape=input_shape)
            else:
                raise ValueError('Invalid weights for feature layer')

            if config['two_gpus']:
                with tf.device('/gpu:0'):
                    img_a_feat = feature_model(img_a_in)
                with tf.device('/gpu:1'):
                    img_b_feat = feature_model(img_b_in)
            else:
                img_a_feat = feature_model(img_a_in)
                img_b_feat = feature_model(img_b_in)
        elif config['input_type'] == 'features':
            img_a_feat = img_a_in
            img_b_feat = img_b_in
        else:
            raise ValueError('Invalid input type for the network.')

        if config['use_triplet_embedding']:
            triplet_embedding_network = TripletEmbeddingNetwork(input_shape, config['triplet_embedding_layer_sizes'], config['triplet_embedding_network_weights'])

            # Freeze weights in the embedding.
            for layer in triplet_embedding_network.layers:
                layer.trainable=False

            img_a_feat = triplet_embedding_network(img_a_feat)
            img_b_feat = triplet_embedding_network(img_b_feat)

        if config['use_distance_network']:
            before_concat_layers = []

            for layer_size in config['distance_network_layer_sizes_before_concat']:
                before_concat_layers.append(KL.Dense(layer_size, activation='relu'))

            for layer in before_concat_layers:
                img_a_feat = layer(img_a_feat)

            for layer in before_concat_layers:
                img_b_feat = layer(img_b_feat)

            distance = KL.Concatenate(axis=1)([img_a_feat, img_b_feat])

            after_concat_layers = []

            for layer_size in config['distance_network_layer_sizes_after_concat']:
                after_concat_layers.append(KL.Dense(layer_size, activation='relu'))

            for layer in after_concat_layers:
                distance = layer(distance)
        else:
            # Set up distance metric
            if config['distance_metric'] == 'l1':
                distance = KL.Lambda(utils.l1_distance)([img_a_feat, img_b_feat])
            elif config['distance_metric'] == 'l2':
                distance = KL.Lambda(utils.l2_distance)([img_a_feat, img_b_feat])
            else:
                raise ValueError('Distance function not supported')

            if mode == 'training' and config['include_reg']:
                distance = BatchNormalization()(distance)
                distance = Activation('relu')(distance)

        distance = KL.Dense(1, activation='sigmoid')(distance)

        keras_model = KM.Model(inputs=(img_a_in, img_b_in),
                            outputs=distance,
                            name='siamese_net')

        if mode == 'inference' or config['weights'] == 'last':
            keras_model.load_weights(config['model_filename'])

        return keras_model

    def predict(self, image_pairs, batch_size=32):
        return self.keras_model.predict(image_pairs, batch_size=batch_size)

    def train(self, config):
        train_dataset = ImageDataset(config['dataset_path'], 'train', config['data_augmentation_suffixes'], config['allow_different_views'])
        train_dataset.prepare(config['num_train_pairs'])

        val_dataset = ImageDataset(config['dataset_path'], 'validation')
        val_dataset.prepare(config['num_val_pairs'])

        train_generator = DataGenerator(train_dataset, batch_size=config['batch_size'],
                                                       dim=self.config['input_shape'],
                                                       shuffle=config['shuffle_training_inputs'],
                                                       dataset_type=config['dataset_type'])
        val_generator = DataGenerator(val_dataset, batch_size=config['batch_size'],
                                                   dim=self.config['input_shape'],
                                                   shuffle=config['shuffle_training_inputs'],
                                                   dataset_type=config['dataset_type'])

        model_path, _ = os.path.split(self.config['model_filename'])
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path, verbose=0, save_weights_only=True)
        ]

        self.keras_model.compile(loss=utils.contrastive_loss,
                                     optimizer=Adam(lr=config['learning_rate']),
                                     metrics=[utils.accuracy, utils.auc_roc, 'acc'])

        history = self.keras_model.fit_generator(generator=train_generator,
                                validation_data=val_generator,
                                epochs=config['epochs'],
                                use_multiprocessing=True,
                                callbacks=callbacks,
                                workers=multiprocessing.cpu_count())


        self.keras_model.save(self.config['model_filename'])
#-------------------------------------------------------

        pred_generator = DataGenerator(val_dataset, batch_size=config['batch_size'],
                                                   dim=self.config['input_shape'],
                                                   shuffle=config['shuffle_training_inputs'],
                                                   dataset_type=config['dataset_type'],
                                                   output=True)
        predictions = self.keras_model.predict_generator(pred_generator,verbose=1)
        np.save('predictions.npy', np.array(predictions))

#-------------------------------------------------------
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc)+1)

        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy %')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig('acc.png')
        plt.figure()

        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Validation %')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig('loss.png')
        plt.show()

#--------------------------------------------------------

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
