"""
Dataset Descriptor:
https://www.nature.com/articles/s41597-022-01560-7
Dataset Link:
https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70225642
"""

#%%
import os
import re
import fnmatch
from glob import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Conv3D, MaxPooling3D  ##!!  
from keras.layers import BatchNormalization
from keras import backend as K
import numpy as np
import nibabel as nib
import pandas as pd
from keras.utils import Sequence
import math
import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
                                

seed1 = 1000  
seed2 = 2000

#%% Pathing
path_source = 'D:\\Projects\\AICC'

# Santinize Path and create sub-paths. 
# if os.name == 'nt':
#     path_source.replace('\\','\\\\')
# path_source = os.path.abspath(path_source)
path_data = os.path.join(path_source,'Data\\UPENN-GBM\\NIfTI-files')
path_code = os.path.join(path_source,'Code')
print(path_source,'\n',path_data,'\n',path_code,'\n')

#%% Get List of directories and validate file names
path_seg = os.path.join(path_data,'images_segm')
path_structural = os.path.join(path_data,'images_structural')
seg_name = os.listdir(path_seg)

sub_names = [x.strip('_segm.nii.gz') for x in seg_name] # List of subjects that have a segmentation file. Generous assumption, but it is assumed that these subjects exist
sub_dirs = [os.path.join(path_structural,x) for x in sub_names]
sub_label = [os.path.join(path_seg,x) for x in seg_name] # Full Paths for all segmentation files. 
print('Subject Directories:', len(sub_dirs))
print('Segmentation Files:',len(sub_label))


# %% Create Training and Test sets
x_total = np.array(sub_dirs) # Paths to all subject structural files
y_total = np.array(sub_label) # Paths to all segmentation files
x_trainT,x_test,y_trainT,y_test=train_test_split(x_total,y_total,test_size=.25,random_state=seed1)
x_train,x_val,y_train,y_val=train_test_split(x_trainT,y_trainT,test_size=.1,random_state=seed2)
#%%
print('Training Set:' + str(x_train.shape))
print('Testing Features:' + str(y_train.shape))
print('Validation Set:' + str(x_val.shape))
print('Validation Features:' + str(y_val.shape))
print('Testing Set:' + str(x_test.shape))
print('Testing Features:' + str(y_test.shape))


# %% Create Generator
class data_generator(Sequence):
    def __init__(self, x_set, y_set, batch_size, target_shape = (240,240,160)):
        self.x, self.y = x_set, y_set
        # self.shape = (batch_size, target_shape[0], target_shape[1], target_shape[2], 4)
        self.batch_size = batch_size
        self.target_shape = target_shape

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        # print(type(low),type(self.batch_size),type(self.x))
        high = min(low + self.batch_size, len(self.x))
        batch_x = self.x[low:high]
        batch_y = self.y[low:high]
        batch_X = []
        for i in range(len(batch_x)):
            nii_file_prefix = batch_x[i]
            # sub_dir = os.listdir(nii_file_prefix)
            # nii_file_prefix = '/common/compbiomed-aicampus/team1/biomed_imaging/Data/PKG_UPENN_GBM_NIfTI_files/NIfTI_files/images_structural/' + batch_x[i] + '/' + batch_x[i]  ##!!  
            nii_T1 = nib.load(os.path.join(nii_file_prefix,glob('*_T1.nii.gz',root_dir = nii_file_prefix)[0])) # Gets the first file found that matches
            W_T1 = nii_T1.get_fdata()
            nii_T2 = nib.load(os.path.join(nii_file_prefix,glob('*_T2.nii.gz',root_dir = nii_file_prefix)[0]))
            W_T2 = nii_T2.get_fdata()
            nii_T1GD = nib.load(os.path.join(nii_file_prefix,glob('*_T1GD.nii.gz',root_dir = nii_file_prefix)[0]))
            W_T1GD = nii_T1GD.get_fdata()
            nii_FLAIR = nib.load(os.path.join(nii_file_prefix,glob('*_FLAIR.nii.gz',root_dir = nii_file_prefix)[0]))
            W_FLAIR = nii_FLAIR.get_fdata()
            # Reshaping early in the batching process for convenience
            W_T1 = resize(W_T1,output_shape = self.target_shape)
            W_T2 = resize(W_T2,output_shape = self.target_shape)
            W_T1GD = resize(W_T1GD,output_shape = self.target_shape)
            W_FLAIR = resize(W_FLAIR,output_shape = self.target_shape)
            # Stacking and appending
            W_all = np.stack([W_T1, W_T2, W_T1GD, W_FLAIR], axis=-1)  ##!!  
            # print(W_all.shape)
            batch_X.append(W_all)
        batch_Y = []
        for i in range(len(batch_y)):
            nii_seg = nib.load(batch_y[i])
            data_seg = nii_seg.get_fdata()
            # Resizing and rounding
            data_seg_resize = resize(data_seg, output_shape = self.target_shape)
            data_seg_resize = np.around(data_seg_resize)
            # Separating by int value
            Y_1s = np.where(data_seg_resize == 1, 1, 0)
            Y_2s = np.where(data_seg_resize == 2, 1, 0)
            Y_4s = np.where(data_seg_resize == 4, 1, 0)
            # Stacking and appending
            Y_all = np.stack([Y_1s, Y_2s, Y_4s], axis=-1)
            batch_Y.append(Y_all)  
        # Reordering
        batch_X = np.stack(batch_X, axis=0)
        batch_Y = np.stack(batch_Y, axis=0)
        
        # Converting to float
        batch_X = np.float32(batch_X)
        batch_Y = np.float32(batch_Y)

        return batch_X, batch_Y # batch_Y_1s, batch_Y_2s, batch_Y_4s
#%% Loss Function (dice score)
def dice_loss(inputs, targets, smooth=1e-6):
    inputs = np.array(inputs); targets = np.array(targets)
    # A separate diceloss function may need to be constructed if multiple classes are present. 
    intersect= np.sum(targets*inputs)
    
    t_sum = np.sum(targets)
    i_sum = np.sum(inputs)
    
    dice = (2 * intersect + smooth) / (t_sum + i_sum + smooth)
    dice = np.mean(dice)
    return 1 - dice

# sci-kit image transform


#%% Just used to test the generator.
# a = data_generator(x_train,y_train, batch_size = 2)
# batch_x, batch_y = a[0]
# data_generator(x_train_fit, y_train_fit, batch_size=batch_size, target_shape = shape_tuple)
# print(batch_y.shape,type(batch_y[0,0,0,0,0]))
# for i in range(len(a)):
#     print('Batch: ', i+1)
#     batch_x, batch_y = a[i]
#     print(batch_x.shape,type(batch_x)) # For troubleshooting
#     print(batch_y.shape,type(batch_y))
#%%
# print(len(x_train),'\n',len(y_train))
# b = [a[x] for x in range(len(a))]

# %% U-Net Architecture (see GBN_unet.py)

from GBN_unet import build_unet_model
start_neurons = 4
## Original Dims (Reshaped in generator)
# xImage = 240; yImage = 240; zImage = 155
xImage = 240
yImage = 240
zImage = 160
shape_tuple = (xImage,yImage,zImage)


# Unet.compile(loss = tf.keras.losses.categorical_crossentropy, optimizer= 'adam', metrics=['accuracy'])
# %%
#Learning Rate Annealer
from keras.callbacks import ReduceLROnPlateau
lrr= ReduceLROnPlateau(monitor='val_accuracy', factor=.001, patience=3, min_lr=1e-5) 

# %% Custom callback to check input_shape
class ShapeCheckCallback(keras.callbacks.Callback):
    def on_train_batch_begin(self, batch_num, logs=None):
        inputs = self.model.inputs
        for i in inputs:
            print('Shape of input:', K.int_shape(i),inputs)

    def on_test_batch_begin(self, batch_num, logs=None):
        inputs = self.model.inputs
        for i in inputs:
            print('Shape of input:', K.int_shape(i),type(inputs[0,0,0,0,0]))

class BatchShapeCallback(keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        # The model's input shape can be obtained by:
        input_shape = self.model.input_shape
        print(f"Input shape at batch {batch}: {input_shape}")
        # The actual batch's shape can be obtained by:
        if self.model.inputs:
            actual_batch_shape = tf.shape(self.model.inputs[0])
            print(f"Actual batch shape at batch {batch}: \n{actual_batch_shape} \n{type(self.model.inputs[0])}")

            
class LayerDataCallback(tf.keras.callbacks.Callback):
    def __init__(self, layer_index):
        super().__init__()
        self.layer_index = layer_index

    def on_train_batch_end(self, batch, logs=None):
        # Get the inputs and outputs of the layer
        inputs = self.model.layers[self.layer_index].input
        outputs = self.model.layers[self.layer_index].output

        # Create a function to return the inputs and outputs
        func = K.function([self.model.input], [inputs, outputs])

        # Use the function to get the inputs and outputs
        layer_inputs, layer_outputs = func([x_train])

        print(f'Inputs to layer {self.layer_index}: {layer_inputs}')
        print(f'Outputs from layer {self.layer_index}: {layer_outputs}')
#%%
#Defining the parameters
batch_size = 4   
num_epochs = 100   
lr = .10
activation = "sigmoid"

tf.keras.backend.clear_session()
Unet = build_unet_model(shape_tuple,start_neurons, lr = lr, activation = activation)
#%% 
# testing on subset of dataset. 
trim_flag = True
if trim_flag:
    x_train_fit = x_train[0:20]
    y_train_fit =  y_train[0:20]
    x_val_fit = x_val[0:8]
    y_val_fit = y_val[0:8]
    x_test_eval = x_test[0:10]
    y_test_eval = y_test[0:10]
else:
    x_train_fit = x_train
    y_train_fit =  y_train
    x_val_fit = x_val
    y_val_fit = y_val
    x_test_eval = x_test
    y_test_eval = y_test
#%%
# Training Unet
train_generator = data_generator(x_train_fit, y_train_fit, batch_size=batch_size, target_shape = shape_tuple)
#%%
unet_1 = Unet.fit(x=train_generator,
         epochs = num_epochs, 
         steps_per_epoch = None, 
         validation_data = data_generator(x_val_fit, y_val_fit, batch_size=batch_size, target_shape = shape_tuple), 
         validation_steps = None, 
         callbacks = [
            #  lrr,
            #  BatchShapeCallback()
             ], 
         verbose=1, 
        #  use_multiprocessing=True
         ) 


#%% Visualize Training Performance

# Initializing subplots
f,ax=plt.subplots(2,1) #Creates 2 subplots under 1 column
plt.tight_layout()

# Training/Validation Loss
ax[0].plot(unet_1.history['loss'],color='b',label='Training Loss')
ax[0].plot(unet_1.history['val_loss'],color='r',label='Validation Loss')
ax[0].plot(np.argmin(unet_1.history["val_loss"]), np.min(unet_1.history["val_loss"]), marker="x", color="r", label="best model")
ax[0].set(xlabel='Epochs',ylabel='Loss')
ax[0].legend()

# # Training/Validation Accuracy
ax[1].plot(unet_1.history['accuracy'],color='b',label='Training  Accuracy')
ax[1].plot(unet_1.history['val_accuracy'],color='r',label='Validation Accuracy')
ax[1].set(xlabel='Epochs',ylabel='Accuracy')
ax[1].legend()





# %% Evaluate the Model on test data
unet_eval = Unet.evaluate(x=data_generator(x_set=x_test_eval, y_set=y_test_eval, batch_size=batch_size, target_shape = shape_tuple), 
            #   use_multiprocessing=True
              )  ##!!  



# %% # Testing output of model
out_gen = data_generator(x_total, y_total, batch_size=1, target_shape = shape_tuple)
#%%
out_x, out_y = out_gen[0]
unet_out = Unet.predict(out_x)
# %%

#%%
print(np.shape(out_x))
cat = 1 # 0-3
frame = 80
plt.figure()
plt.imshow(unet_out[0,0:239,0:239,frame,cat])
plt.figure()
plt.imshow(out_y[0,0:239,0:239,frame,cat])
# %%
