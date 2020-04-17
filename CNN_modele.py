# -*- coding: utf-8 -*-
#!/usr/bin/env python3

# Librairies
from import_data import *
from train_model import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow.python.keras.layers import Input, Dense, Conv2D, Flatten, MaxPool2D, BatchNormalization,Dropout,MaxPooling2D
from tensorflow.python.keras.models import Model,Sequential
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import *
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.datasets import mnist
from keras.utils import to_categorical, normalize
from tensorflow.python.keras.callbacks import LearningRateScheduler
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

#Modele_CNN
# Initialisation
model = Sequential()

# ------------------------------------
# Couche de Convolution et MaxPooling
# ------------------------------------

# Conv2D :
#     filters : nombres de filtres de convolutions
#     kernel_size : taille des filtres de la fenêtre de convolution
#     input_shape : taille de l'image en entrée (à préciser seulement pour la première couche)
#     activation  : choix de la fonction d'activation
# BatchNormalisation : permet de normaliser les coefficients d'activation afin de les maintenirs proche de 0 pour simplifier les calculs numériques
# MaxPooling : Opération de maxPooling sur des données spatiales (2D)
# Dropout : permet de désactiver aléatoirement une proportion de neurones (afin d'éviter le surentrainement sur le jeu d'entrainement)

model.add(Conv2D(filters = 32, kernel_size = (5, 5), activation='relu', padding='Same', input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (5, 5), activation='relu', padding='Same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(strides=(2,2)))
model.add(Dropout(0.25))

# Flatten : conversion d'une matrice en un vecteur plat
# Dense   : neurones
model.add(Flatten())     # Applatissement de la sortie du réseau de convolution
model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(0.25))
# Couche de sortie : nombre de neurones = nombre de classe à prédire
model.add(Dense(units=10, activation='softmax'))

# Récapitulatif de l'architecture modèle
model.summary()

# Sélection de l'optimiser pour le gradient descent
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
