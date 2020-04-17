# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import numpy as np
from sklearn import preprocessing
import keras
from keras.utils import to_categorical, normalize

#Normaliser les données et encoder les labels pour qu'ils prennent la forment
#d'une distribution de probabilité sur les classes (one hot encoding).

def features_normalization (X):
    #version Keras (bug)
    x_normalized =  keras.utils.normalize(X, axis=-1, order=2)
    #version numpy
    #x_normalized = preprocessing.normalize(X)
    return x_normalized

def labels_encoder(Y,n):
    y_encoder=keras.utils.to_categorical(Y, num_classes=n,dtype='float32')
    return y_encoder
