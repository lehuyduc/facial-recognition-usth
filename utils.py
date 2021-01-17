# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 10:00:06 2021

@author: lehuy
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
import sklearn.metrics as metrics
import pandas as pd
import cv2
import itertools


def readInput():
    df = pd.read_csv('fer2013/fer2013.csv')
    df.head()
    
    #### Reading data from dataset
    image_size=(48,48)
    input_shape = (48,48)
    pixels = df['pixels'].tolist() # Converting the relevant column element into a list for each row
    width, height = 48, 48
    faces = []
    
    for pixel_sequence in pixels:
      face = [int(pixel) for pixel in pixel_sequence.split(' ')] # Splitting the string by space character as a list
      face = np.asarray(face).reshape(width, height) #converting the list to numpy array in size of 48*48
      face = cv2.resize(face.astype('uint8'),image_size) #resize the image to have 48 cols (width) and 48 rows (height)
      faces.append(face.astype('float32')) #makes the list of each images of 48*48 and their pixels in numpyarray form
      
    faces = np.asarray(faces) #converting the list into numpy array
    faces = np.expand_dims(faces,-1) #Expand the shape of an array -1=last dimension => means color space
    emotions = pd.get_dummies(df['emotion']).to_numpy() #doing the one hot encoding type on emotions
    
    ####-------------    
    x_total = faces.astype('float32')
    y_total = emotions    
    
    return x_total, y_total


def plotLoss(val_loss, train_loss):
    plt.figure()
    plt.plot(val_loss, color='green', label='Validation error')
    plt.plot(train_loss, color='blue', linestyle='--', label='Training error')
    plt.xlabel('Epochs')
    plt.title('History of training loss and validation loss')
    plt.legend()
  
def plotAccuracy(val_accuracy, train_accuracy):
    plt.figure()
    plt.plot(val_accuracy, color='green', label='Validation accuracy')
    plt.plot(train_accuracy, color='blue', linestyle='--', label='Training accuracy')
    plt.xlabel('Epochs')
    plt.title('History of training accuracy and validation accuracy')
    plt.legend()
  
def plotConfusionMatrix(classes, y_truth, y_pred, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = sklearn.metrics.confusion_matrix(y_truth, y_pred, labels=classes)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title if not normalize else title+' (normalized)')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        value = cm[i, j] if not normalize else '{:.2f}%'.format(100*cm[i, j])
        plt.text(j, i, value, horizontalalignment='center', color='red' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
