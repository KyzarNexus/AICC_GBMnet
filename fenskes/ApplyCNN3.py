#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:19:10 2023

@author: fenskes
"""

#Goal is to train a model on imaging datsets T1, T2, T1Gd, and Flair capable of predicting MGMT classification: methylated (better prognosis), intermediate, and unmethylated
#import libraries using tensorflow
#pip install tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd

#import Nibabel libraries
#pip install nibabel
import os
import numpy as np
import nibabel as nib

#Import keras 
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization #doesn't work

#Onehot encoding of MGMT data
import sklearn
from sklearn.utils.multiclass import unique_labels
from keras.utils import to_categorical

#further split training data into traning and validation
import sklearn
from sklearn.model_selection import train_test_split


#start with training split
#load subject participant and MGMT dataset
clin_dat= pd.read_csv('/Users/fenskes/Library/CloudStorage/Box-Box/AISchool/data/TrainTestSplit/ClinicalInfo_MGMT_Train.csv', sep='\t')

#split dataset into training and validation - 10% of the total
clin_dat_train,clin_dat_val=train_test_split(clin_dat,test_size=.1)
print((clin_dat_train.shape))
print((clin_dat_val.shape))

#MGMT as a categorical using one-hot encoding
y_train = to_categorical(np.asarray(clin_dat_train.MGMT.factorize()[0]))
y_val = to_categorical(np.asarray(clin_dat_val.MGMT.factorize()[0]))
print((y_train.shape))
print((y_val.shape))

#Subject ID
subj_train = clin_dat_train.ID
subj_val = clin_dat_val.ID


#work with datasets in batches/multiprocessing using keras.utils.sequence

#start with training dataset
i = 0
#x_train = np.zeros((subj_train.shape[0],240,240,155,4)) #if working on whole dataset
x_train = np.zeros((20,240,240,155,4)) 
for x in subj_train:

#load datset using NiBabel
    data_path = '/Users/fenskes/Desktop/HPC/common/compbiomed-aicampus/team1/biomed_imaging/Data/PKG_UPENN_GBM_NIfTI_files/NIfTI_files/images_structural/' + x
    FLAIR_Data = os.path.join(data_path, x + '_FLAIR.nii.gz')
    T1_Data = os.path.join(data_path, x + '_T1.nii.gz')
    T1GD_Data = os.path.join(data_path, x + '_T1GD.nii.gz')
    T2_Data = os.path.join(data_path, x + '_T2.nii.gz')

    FLAIR_img = nib.load(FLAIR_Data)
    T1_img = nib.load(T1_Data)
    T1GD_img = nib.load(T1GD_Data)
    T2_img = nib.load(T2_Data) 

#convert to array datasets
    FLAIR_image_data = FLAIR_img.get_fdata()  
    T1_image_data = T1_img.get_fdata()
    T1GD_image_data = T1GD_img.get_fdata()
    T2_image_data = T2_img.get_fdata() 

#normalize values between 0 and 1 
    FLAIR_image_dataN = (FLAIR_image_data-np.min(FLAIR_image_data)/(np.max(FLAIR_image_data))-np.min(FLAIR_image_data))
    T1_image_dataN = (T1_image_data-np.min(T1_image_data)/(np.max(T1_image_data))-np.min(T1_image_data))
    T1GD_image_dataN =(T1GD_image_data-np.min(T1GD_image_data)/(np.max(T1GD_image_data))-np.min(T1GD_image_data))
    T2_image_dataN = (T2_image_data-np.min(T2_image_data)/(np.max(T2_image_data))-np.min(T2_image_data)) 

    x_train0 = np.stack([[FLAIR_image_dataN], [T1_image_dataN], [T1GD_image_dataN], [T2_image_dataN]], axis=4)

    x_train[i,:,:,:,:] = x_train0
        
    i = i+1

#train on a smaller dataset first to test code!
    if x == 'UPENN-GBM-00294_11':
        break

#make sure the variable for y_train is the right length (using the corresponding MGMT values)
y_train = y_train[0:x_train.shape[0]-1]

#validation dataset (same process as training)
#start with training dataset

i = 0
#x_val= np.zeros((subj_val.shape[0],240,240,155,4)) #if working on whole dataset
x_val= np.zeros((2,240,240,155,4))
for x in subj_val:

#load datset using NiBabel
    data_path = '/Users/fenskes/Desktop/HPC/common/compbiomed-aicampus/team1/biomed_imaging/Data/PKG_UPENN_GBM_NIfTI_files/NIfTI_files/images_structural/' + x
    FLAIR_Data = os.path.join(data_path, x + '_FLAIR.nii.gz')
    T1_Data = os.path.join(data_path, x + '_T1.nii.gz')
    T1GD_Data = os.path.join(data_path, x + '_T1GD.nii.gz')
    T2_Data = os.path.join(data_path, x + '_T2.nii.gz')

    FLAIR_img = nib.load(FLAIR_Data)
    T1_img = nib.load(T1_Data)
    T1GD_img = nib.load(T1GD_Data)
    T2_img = nib.load(T2_Data) 

#convert to array datasets
    FLAIR_image_data = FLAIR_img.get_fdata()  
    T1_image_data = T1_img.get_fdata()
    T1GD_image_data = T1GD_img.get_fdata()
    T2_image_data = T2_img.get_fdata() 

#normalize values between 0 and 1 
    FLAIR_image_dataN = (FLAIR_image_data-np.min(FLAIR_image_data)/(np.max(FLAIR_image_data))-np.min(FLAIR_image_data))
    T1_image_dataN = (T1_image_data-np.min(T1_image_data)/(np.max(T1_image_data))-np.min(T1_image_data))
    T1GD_image_dataN =(T1GD_image_data-np.min(T1GD_image_data)/(np.max(T1GD_image_data))-np.min(T1GD_image_data))
    T2_image_dataN = (T2_image_data-np.min(T2_image_data)/(np.max(T2_image_data))-np.min(T2_image_data)) 


    x_val0 = np.stack([[FLAIR_image_dataN], [T1_image_dataN], [T1GD_image_dataN], [T2_image_dataN]], axis=4)

    x_val[i,:,:,:,:] = x_val0
        
    i = i+1

#train on a smaller dataset first to test code!
    if x == 'UPENN-GBM-00336_11':
        break

#make sure the variable for y_train is the right length (using the corresponding MGMT values)
y_val = y_val[0:x_val.shape[0]-1]

#

#Need to save images as x_train data - how do you input each channel?

#set up CNN (adjusted from 2D https://analyticsindiamag.com/hands-on-guide-to-implementing-alexnet-with-keras-for-multi-class-image-classification/)
AlexNet = Sequential()

#1st convolutional layer
AlexNet.add(Conv3D(filters=96, input_shape=(240,240,155,4)), kernel_size=(11,11,11), strides=(4,4,4),padding='same')
AlexNet.add(BatchNormalization())
AlexNet.add(Ativation('relu'))
AlexNet.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='same'))

#2nd convolutional layer
AlexNet.add(Conv3D(filters=256, kernel_size=(5,5,5), strides=(1,1,1), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='same'))

#3rd convolutional layer
AlexNet.add(Conv3D(filters=384, kernel_size=(3,3,3), strides=(1,1,1), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))

#4th convolution layer
AlexNet.add(Conv3D(filters=384, kernel_size(3,3,3),strides(,1,1,), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))

#5th convolutional layer
AlexNet.add(conv2D(filters=256, kernel_size=(3,3,3), strides=(1,1,1), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activiation('relu'))
AlexNet.add(MaxPooling3(pool_size=(2,2,2), strides=(2,2,2), padding='same'))

#passing it to a fully connected layer
AlexNet.add(Flatten())
#1st fully connected layer
AlexNet.add(Dense(4096, input_shaped=(240,240,155,4)))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
#add dropout to prevent overfitting
AlexNet.add(Dropout(0.4))

#2nd fully connected layer
AlexNet.add(Dense(4096))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
#add dropout
AlexNet.add(Dropout(0.4))

#3rd fully connected layer
AlexNet.add(Dense(1000))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
#add dropout
AlexNet.add(Dropout(0.4))

#output layer
AlexNet.add(Dense(10))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('softmax'))

#model summary
AlexNet.summary()
AlexNet.compile(loss = keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

#IS THIS NECESSARY??
#Image Data Augmentation
from keras.preprocessing.image import ImageDataGenerator

train_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True,zoom_range=.1 )
val_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True,zoom_range=.1)
test_generator = ImageDataGenerator(rotation_range=2, horizontal_flip= True,zoom_range=.1)

#Fitting the augmentation defined above to the data
train_generator.fit(x_train)
val_generator.fit(x_val)
test_generator.fit(x_test)

#learning rate annealer
from keras.callbacks
import ReduceLROnPlateau
llr = ReduceLROnPlateau(mpnitor='val_acc', factor-0.1, patience=3, min_lr=1e-5)

#define the parameters
batch_size = 100
epochs = 100
learn_rate = 0.001

#Training the model
#tensorflow version is 2.12 so it's okay to use fit_generator (otherwise, if 2.2 use fit)
AlexNet.fit_generator(train_generator.flow(x_train,y_train batch_size=batch_size), epochs = epochs, spteps_per_epoch = x_train.shape[0]//batch_size, validation_data = val_generator.flow(x_val, y_val, batch_size=batch_size), validation_steps = 250, calbacks = [lrr], verbose=1)

#plot training and validation
import matplotlib.pyplot as plt
f,ax=plt.subplots(2,1) 

#assign first subplot to graph training loss and validation loss
ax[0].plot(AlexNet.history.history['loss'], color='b', label='Training Loss')
ax[0].plot(AlexNet.history.history['val_loss'], color='r', label='Validation Loss')

#ploting the training accuracy and validation accuracy
ax[1].plot(AlexNet.history.history['accuracy'], color='b', label='Training Accuracy')
ax[1[.plot(AlexNet.hisotry.history['val_accuracy'], color='r', label='Validation Accuracy')
     
plt.legend()

#Defining function for confusion matrix plot
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, tital=None, cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
            
#Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
if normalize:
    cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    print(("Normalized confusion matrix")
else:
    print('Confusion matrix, without normalization')
    
    
#Print Confusion matrix
fig, ax = plt.subplots(figsize = (7,7))
im = ax.imshow(cm, interpolation='nearest',cmap=cmap)
ax.figure.colorbar(im, ax=ax)
#We want to show all ticks...
ax.set(xticks=np.arange(cm.shap[1])), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes, title=title, ylabel='True label', xlabel='Predicted label')
        
#Rotate the tick labels and set their alignment
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#Loop over data dimensions and create text annotations.
fmt = '.2f' if normalize else 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i,j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else 'black')
fig.tight_layout()
return ax

np.set_printoptions(precision=2)

#Making the prediction
#load x_test and y_test from split data
#Fitting the augmentation defined above to the data
test_generator.fit(x_test)


y_pred=AlexNet.predict_classes(x_test)
y_true=np.argmax(y_test, axis=1)

#plotting the confusion matrix
from sklearn.metrics import confusion_matrix
confusion_mtx=confusion_matrix(y_true,y_pred)

class_names=['Unmethylated', 'Indeterminate', 'Methylated']

#plotting non-normalized confusion matrix
plot_confusion_matrix(y_true, y_pred, classes = class_names, title = 'Confusion matrix without nomralization')

#plotting normalized confusion matrix
plot_confusion_matrix(u_true, y_pred, classes=class_names, normalize=True, title='Normalized confusion matrix')

#Classification accuracy
from sklearn.metrics import accuracy_score
acc_score = acuracy_score(y_true, y_pred)
print('Accuracy Score = ',acc_score)


    







