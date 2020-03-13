# -*- coding: utf-8 -*-
import os
import numpy as np
import cPickle as pickle
import theano
import sys
import fnmatch
from PIL import Image
from lasagne import layers
from lasagne import objectives
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
from lasagne.objectives import categorical_crossentropy
from lasagne import nonlinearities
from scipy import ndimage

early_stopping = EarlyStopping(patience = 100)

def get_CNN4():
    CNN4 = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),  # !
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('dropout2', layers.DropoutLayer),  # !
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('dropout3', layers.DropoutLayer),  # !
            ('conv4', layers.Conv2DLayer),
            ('pool4', layers.MaxPool2DLayer),
            ('dropout4', layers.DropoutLayer),  # !
            ('hidden5', layers.DenseLayer),
            ('dropout5', layers.DropoutLayer),  # !
            ('hidden6', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 384, 384),
        conv1_num_filters=16, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
        dropout1_p=0.1,
        conv2_num_filters=32, conv2_filter_size=(3, 3), pool2_pool_size=(2, 2),
        dropout2_p=0.2,
        conv3_num_filters=64, conv3_filter_size=(3, 3), pool3_pool_size=(2, 2),
        dropout3_p=0.3,
        conv4_num_filters=128, conv4_filter_size=(3, 3), pool4_pool_size=(2, 2),
        dropout4_p=0.5,
        hidden5_num_units=300,
        dropout5_p=0.5,
        hidden6_num_units=300,
        output_num_units=2, output_nonlinearity=nonlinearities.softmax,
        update_learning_rate=theano.shared(float32(0.001)),
        update_momentum=theano.shared(float32(0.9)),
        regression=False,
        objective_loss_function = categorical_crossentropy,
        batch_iterator_train=BatchIterator(batch_size=48),
        on_epoch_finished=[early_stopping],
        on_training_finished=[LoadBestWeight(early_stopping)],
        max_epochs=100,
        verbose=2,
        )
    return CNN4

class EarlyStopping(object):

    def __init__(self, patience = 100):
       self.patience = patience
       self.best_valid = np.inf
       self.best_valid_epoch = 0
       self.best_valid_accuracy = 0
       self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        current_valid_accuracy = train_history[-1]['valid_accuracy']
        if current_valid_accuracy > self.best_valid_accuracy:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_valid_accuracy = current_valid_accuracy
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            if nn.verbose:
                print("Best valid loss was {:.6f} at epoch {} with accuracy {}.".format(
                    self.best_valid, self.best_valid_epoch, self.best_valid_accuracy))
            nn.load_params_from(self.best_weights)
            if nn.verbose:
                print("Weights set.")
            raise StopIteration()

class LoadBestWeight:
    def __init__(self, early_stopping):
        self.early_stopping = early_stopping

    def __call__(self, nn, train_history):
        print("Training stage finishes. Best valid loss was {:.6f} at epoch {} with accuracy {}.".format(
            self.early_stopping.best_valid, self.early_stopping.best_valid_epoch, \
                self.early_stopping.best_valid_accuracy))
        nn.load_params_from(self.early_stopping.best_weights)

def float32(k):
    return np.cast['float32'](k)
