# -*- coding: utf-8 -*-
#!/usr/bin/env python3

# Librairies
from import_data import *
from train_model import *
from CNN_modele import *
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

# Dataframe train
df_train = importation_file("/Users/Ricou/Desktop/ANDRE/machine_learning/data/train.csv")
print("df_train shape",df_train.shape)

#Séparation des features et des Labels
#Labels
y_train=df_train['label']
#features
#28*28 = 784 correspond à la taille et la largeur de chaque image
X_train = df_train.drop(labels = ["label"],axis = 1)

# Split entre jeu d'entrainement et jeu de validation avec un ratio de 90/10
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=False)

#Shape
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

# Reshape des data pour les formater en 28x28x1
X_train = X_train.values.reshape(-1, 28,28,1)
X_val = X_val.values.reshape(-1, 28,28,1)

#Normalisation des features
X_train = normalize(X_train)
X_val = normalize(X_val)

#Recherche du nombre de classes présentes dans les labels
n = len(np.unique(y_train))

# Encodage en One hot labels (de 0 a 9)
y_train = labels_encoder(y_train,n)
y_val = labels_encoder(y_val,n)
print(y_train.shape,y_val.shape)

    #Modele_CNN
#Entrainement du modèle
Model = model.fit (X_train, y_train, validation_data = (X_val, y_val), epochs = 10)

loss, acc = model.evaluate(X_val, y_val, verbose=0)
print(" loss: {0:.4f}, accuracy: {1:.4f}".format(loss, acc))

#Test Dataframe
df_test = importation_file("/Users/Ricou/Desktop/ANDRE/machine_learning/data/test.csv")

# Traitement des données de la même façon que pour l'entrainement
# Reshape
X_test = X_test.values.reshape(-1, 28,28,1)
# Normalisation
X_test = normalize(X_test)

# Prédictions sur le jeu de test
y_hat = model.predict(X_test, verbose=1)
y_pred = np.argmax(y_hat, axis=1)

# Soumission et Enregistrement des résultats
results = pd.Series(y_pred, name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("/Users/Ricou/Desktop/ANDRE/machine_learning/data/nn_mnist_predictions.csv",index=False)
