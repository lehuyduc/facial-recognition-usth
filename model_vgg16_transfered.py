# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:32:42 2021

@author: lehuy
"""

import numpy as np
import pandas as pd
import keras, keras.backend, keras.utils.np_utils, keras.preprocessing.image
from keras import layers, models
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten,Activation, MaxPool2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from utils import readInput, plotLoss, plotAccuracy, plotConfusionMatrix
import pickle

#  (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
class_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

#### read input
x_total, y_total = readInput()
num_classes = np.shape(y_total)[1]
input_shape = (48,48,1)

#### normalize input feature to range -1..1
x_total = ((x_total / 255.0) - 0.5) * 2.0  

#### IMPORTANT: VGG16 requires RGB input -> convert them
gray = np.reshape(x_total, (-1, 48, 48))
x_total = np.stack((gray,)*3, axis=-1)

### split train/valid/test data
x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size=0.2, stratify=y_total, random_state=42)
x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5, stratify=y_test, random_state=42)
print('training classes: ' + str([np.sum(np.argmax(y_train,axis=1)==i) for i in range(num_classes)]))
print('validation classes: ' + str([np.sum(np.argmax(y_valid,axis=1)==i) for i in range(num_classes)]))
print('test classes: ' + str([np.sum(np.argmax(y_test,axis=1)==i) for i in range(num_classes)]))
    

y_train_label = np.argmax(y_train, axis=1)
balance = np.array([1 / (np.sum(y_train_label==i) / len(y_train)) for i in range(num_classes)])
balance = np.array([balance[i] / np.sum(balance) for i in range(num_classes)])
class_weights = {i: balance[i] for i in range(num_classes)}

####----
#### function to get base model

def VggModel():    
    cnn_base = VGG16(include_top=False, input_shape=(48, 48, 3))
    cnn_base.trainable = True
    
    flat = Flatten()(cnn_base.layers[-1].output) # connect last layer of cnn base to flatten
    dense = Dense(128, activation='relu')(flat)
    drop = Dropout(0.3)(dense)
    output = Dense(7, activation='softmax')(drop)
    model = Model(inputs=cnn_base.inputs, outputs=output)
    return model

model = VggModel()
model.summary()

### create data generator
datagen = ImageDataGenerator(                        
                        rotation_range=4,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1)

batch_size = 32
train_batches = datagen.flow(x_train, y_train, batch_size=batch_size)    
valid_batches = datagen.flow(x_valid, y_valid, batch_size=batch_size)
test_batches = datagen.flow(x_test, y_test, batch_size=batch_size)

##### parameters for the model compilation + training
opt = keras.optimizers.Adam(learning_rate=1e-5)
    
early_stop = EarlyStopping('val_loss', patience=7)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.5,patience=2,cooldown=2)
num_epochs = 20

#---
### train model without data augmentation
model = VggModel()
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model_checkpoint1 = ModelCheckpoint('vgg_model.h5', 'val_loss',save_best_only=True)
my_callbacks1 = [model_checkpoint1, early_stop, reduce_lr]
history1 = model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_valid,y_valid), epochs=num_epochs, 
                    callbacks=my_callbacks1)

with open('vgg_history.pkl','wb') as f:
    pickle.dump(history1.history, f)
    
###
model.evaluate(x_test, y_test)
 
val_loss=(history1.history['val_loss'])
train_loss=(history1.history['loss'])
plotLoss(val_loss, train_loss)    

val_accuracy = history1.history['val_accuracy']
train_accuracy = history1.history['accuracy']
plotAccuracy(val_accuracy, train_accuracy)

y_pred = model.predict(x_test)
y_test_label = [class_labels[i] for i in np.argmax(y_test,axis=1)]
y_pred_label = [class_labels[i] for i in np.argmax(y_pred,axis=1)]
plotConfusionMatrix(class_labels,y_test_label, y_pred_label, title='Confusion matrix - vgg16')


#---
### train model with data augmentation
opt2 = keras.optimizers.Adam(learning_rate=1e-5)
model_augmented = VggModel()
model_augmented.compile(optimizer=opt2, loss='categorical_crossentropy', metrics=['accuracy'])
model_checkpoint2 = ModelCheckpoint('vgg_model_augmented.h5', 'val_accuracy',save_best_only=True)
my_callbacks2 = [model_checkpoint2, early_stop, reduce_lr]
history2 = model_augmented.fit(train_batches,
                               steps_per_epoch = len(x_train) // batch_size, 
                               validation_data = valid_batches,
                               validation_steps = len(x_valid) // batch_size,
                               epochs  = num_epochs,
                               callbacks = my_callbacks2)       
                  
with open('vgg_history_augmented.pkl','wb') as f:
    pickle.dump(history2.history, f)
    
###
model_augmented.evaluate(test_batches)

val_loss=(history2.history['val_loss'])
train_loss=(history2.history['loss'])
plotLoss(val_loss, train_loss)    

val_accuracy = history2.history['val_accuracy']
train_accuracy = history2.history['accuracy']
plotAccuracy(val_accuracy, train_accuracy)

y_pred = model_augmented.predict(x_test)
y_test_label = [class_labels[i] for i in np.argmax(y_test,axis=1)]
y_pred_label = [class_labels[i] for i in np.argmax(y_pred,axis=1)]
plotConfusionMatrix(class_labels,y_test_label, y_pred_label, title='Confusion matrix - vgg16 augmented')



#######
model = keras.models.load_model('vgg_model.h5')
history1 = pickle.load(open('vgg_history.pkl','rb'))

model.evaluate(x_test, y_test)
    
val_loss=(history1['val_loss'])
train_loss=(history1['loss'])
plotLoss(val_loss, train_loss)    

val_accuracy = history1['val_accuracy']
train_accuracy = history1['accuracy']
plotAccuracy(val_accuracy, train_accuracy)


###
model_augmented = keras.models.load_model('vgg_model_augmented.h5')
history2 = pickle.load(open('vgg_history_augmented.pkl','rb'))

model_augmented.evaluate(x_test, y_test)
    
val_loss=(history2['val_loss'])
train_loss=(history2['loss'])
plotLoss(val_loss, train_loss)    

val_accuracy = history2['val_accuracy']
train_accuracy = history2['accuracy']
plotAccuracy(val_accuracy, train_accuracy)
