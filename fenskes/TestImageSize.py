#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 10:35:17 2023

@author: fenskes
"""
#pip! install tensorflow
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
from keras.layers.normalization import BatchNormalization

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

    print(FLAIR_img.shape)
    print(T1_img.shape)
    print(T1GD_img.shape)
    print(T2_img.shape)

