#%%
import os
import re
import fnmatch
# from glob import glob
from pathlib import Path
# from sklearn.model_selection import train_test_split
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv3D, MaxPooling3D  ##!!  
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D  ##!!  
from keras.layers import BatchNormalization
import numpy as np
import nibabel as nib
import pandas as pd
from keras.utils import Sequence
import math

# from tqdm import tqdm_notebook, tnrange
# from itertools import chain
# from skimage.io import imread, imshow, concatenate_images
# from skimage.transform import resize
# from skimage.morphology import label
# from sklearn.model_selection import train_test_split

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Concatenate, Reshape
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Conv3DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#%% Loss Function (dice score)
def dice_loss(inputs, targets, smooth=1e-6):
    # inputs = np.array(inputs); targets = np.array(targets) # # A separate diceloss function may need to be constructed if multiple classes are present. 
    # intersect= np.sum(targets*inputs)
    # t_sum = np.sum(targets)
    # i_sum = np.sum(inputs)
    # dice = (2 * intersect + smooth) / (t_sum + i_sum + smooth)
    # dice = np.mean(dice)
    y_true = inputs
    y_pred = targets
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1 - dice


    
    
    
#%%
def build_unet_model(shape_tuple = (240, 240, 160), start_neurons = 4):
    # Constructing U-Net (4 downsteps)
    # General Structure from source 1:
    # https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5
    # Modifications from source 2:
    # https://arxiv.org/pdf/1606.06650.pdf
    
    # Note: 
    # Conv + upsampling
    true_shape = (shape_tuple[0], shape_tuple[1], shape_tuple[2], 4)
    # Input Layer
    input_layer = Input(shape = true_shape, name = 'img')
    # input_layer = BatchNormalization()(input_layer)

    # Down-Step 1    
    conv1 = Conv3D(start_neurons*1,(3, 3, 3),activation='relu',padding='same')(input_layer)
    conv1 = Conv3D(start_neurons*1,(3, 3, 3),activation='relu',padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=2, strides=2, padding = 'same')(conv1)
    pool1 = Dropout(0.25)(pool1)
    # Down-Step 2
    conv2 = Conv3D(start_neurons*2,(3, 3, 3),activation='relu',padding='same')(pool1)
    conv2 = Conv3D(start_neurons*2,(3, 3, 3),activation='relu',padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=2, strides=2, padding = 'same')(conv2)
    pool2 = Dropout(0.5)(pool2)
    # Down-Step 3
    conv3 = Conv3D(start_neurons*4,(3, 3, 3),activation='relu',padding='same')(pool2)
    conv3 = Conv3D(start_neurons*4,(3, 3, 3),activation='relu',padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling3D(pool_size=2, strides=2, padding = 'same')(conv3)
    pool3 = Dropout(0.5)(pool3)
    # Down-Step 4
    conv4 = Conv3D(start_neurons*8,(3, 3, 3),activation='relu',padding='same')(pool3)
    conv4 = Conv3D(start_neurons*8,(3, 3, 3),activation='relu',padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling3D(pool_size=2, strides=2, padding = 'same')(conv4)
    pool4 = Dropout(0.5)(pool4)
    # Mid 
    convm = Conv3D(start_neurons*16, (3, 3, 3), activation="relu", padding="same")(pool4)
    convm = Conv3D(start_neurons*16, (3, 3, 3), activation="relu", padding="same")(convm)
    convm = BatchNormalization()(convm)
    # Up-Step 4
    deconv4 = Conv3DTranspose(start_neurons * 8, (3, 3, 3), strides=(2, 2, 2), padding="same")(convm)
    uconv4 = Concatenate()([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv3D(start_neurons*8, (3, 3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv3D(start_neurons*8, (3, 3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = BatchNormalization()(uconv4)

    # Up 3
    deconv3 = Conv3DTranspose(start_neurons * 4, (3, 3, 3), strides=(2, 2, 2), padding="same")(uconv4)
    uconv3 = Concatenate()([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv3D(start_neurons*4, (3, 3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv3D(start_neurons*4, (3, 3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = BatchNormalization()(uconv3)
    
    # Up 2
    deconv2 = Conv3DTranspose(start_neurons * 2, (3, 3, 3), strides=(2, 2, 2), padding="same")(uconv3)
    uconv2 = Concatenate()([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv3D(start_neurons*2, (3, 3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv3D(start_neurons*2, (3, 3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = BatchNormalization()(uconv2)
    # Up 1
    deconv1 = Conv3DTranspose(start_neurons * 1, (3, 3, 3), strides=(2, 2, 2), padding="same")(uconv2)
    uconv1 = Concatenate()([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv3D(start_neurons * 1, (3, 3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv3D(start_neurons * 1, (3, 3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    # Output
    # 1st entry, numbers of classes
    # activation may need to be 'linear' or 'softmax'. 
    # Might be 3 classes??
    output_layer = Conv3D(3, (1,1,1), padding="same", activation="softmax")(uconv1)

    model = Model(input_layer,output_layer)
    # Summary
    model.summary()
    model.compile(loss = dice_loss, optimizer= 'adam', metrics=['accuracy'])
    return model


#%%
if __name__ == '__main__':
    start_neurons = 4
    xImage = 240
    yImage = 240
    zImage = 160
    shape_tuple = (xImage,yImage,zImage,1)
    unet = build_unet_model(shape_tuple,start_neurons)
    

    





# %%
