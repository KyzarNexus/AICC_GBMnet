# %%
import socket
print(socket.gethostname())

# %%
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv3D, MaxPooling3D  ##!!  
from keras.layers import BatchNormalization
import numpy as np
import nibabel as nib
import pandas as pd
from keras.utils import Sequence
import math

random_seed = 1000  ##!!  

#y_label = ['Male', 'Female']  ##!!  
y_label = ['Unmethylated', 'Indeterminate', 'Methylated']  ##!!  
num_classes = len(y_label)  ##!!  
csv_MGMT_train = pd.read_csv('/common/chenh1/AI_Campus/demographic/ClinicalInfo_MGMT_Train.csv', sep = '\t')  ##!!  
ID_train = csv_MGMT_train['ID']
len_train = len(ID_train)
Gender_train = np.array([0 if csv_MGMT_train['Gender'][i]=='M' else 1 for i in range(len_train)])
MGMT_train = np.zeros(len_train, np.int8)
for i in range(len_train):
    if csv_MGMT_train['MGMT'][i] == 'Indeterminate':
        MGMT_train[i] = 1
    elif csv_MGMT_train['MGMT'][i] == 'Methylated':
        MGMT_train[i] = 2
x_train = np.array(ID_train)  ##!!  
y_train = np.expand_dims(MGMT_train, axis=-1)  ##!!  
csv_MGMT_test = pd.read_csv('/common/chenh1/AI_Campus/demographic/ClinicalInfo_MGMT_Test.csv', sep = '\t')  ##!!  
ID_test = csv_MGMT_test['ID']
len_test = len(ID_test)
Gender_test = np.array([0 if csv_MGMT_test['Gender'][i]=='M' else 1 for i in range(len_test)])
MGMT_test = np.zeros(len_test, np.int8)
for i in range(len_test):
    if csv_MGMT_test['MGMT'][i] == 'Indeterminate':
        MGMT_test[i] = 1
    elif csv_MGMT_test['MGMT'][i] == 'Methylated':
        MGMT_test[i] = 2
x_test = np.array(ID_test)  ##!!  
y_test = np.expand_dims(MGMT_test, axis=-1)  ##!!  

#Train-validation-test split
from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=.3,random_state=random_seed)

#Dimension of the T1 dataset
print((x_train.shape,y_train.shape))
print((x_val.shape,y_val.shape))
print((x_test.shape,y_test.shape))

#Onehot Encoding the labels.
from sklearn.utils.multiclass import unique_labels
from keras.utils import to_categorical

#Since we have 3 classes we should expect the shape[1] of y_train,y_val and y_test to change from 1 to 3
y_train=to_categorical(y_train, num_classes)
y_val=to_categorical(y_val, num_classes)
y_test=to_categorical(y_test, num_classes)

#Verifying the dimension after one hot encoding
print((x_train.shape,y_train.shape))
print((x_val.shape,y_val.shape))
print((x_test.shape,y_test.shape))


class data_generator(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, len(self.x))
        batch_x = self.x[low:high]
        batch_y = self.y[low:high]
        batch_X = []
        for i in range(len(batch_x)):
            nii_file_prefix = '/common/compbiomed-aicampus/team1/biomed_imaging/Data/PKG_UPENN_GBM_NIfTI_files/NIfTI_files/images_structural/' + batch_x[i] + '/' + batch_x[i]  ##!!  
            nii_T1 = nib.load(nii_file_prefix + '_T1.nii.gz')
            W_T1 = nii_T1.get_fdata()
            nii_T2 = nib.load(nii_file_prefix + '_T2.nii.gz')
            W_T2 = nii_T2.get_fdata()
            nii_T1GD = nib.load(nii_file_prefix + '_T1GD.nii.gz')
            W_T1GD = nii_T1GD.get_fdata()
            nii_FLAIR = nib.load(nii_file_prefix + '_FLAIR.nii.gz')
            W_FLAIR = nii_FLAIR.get_fdata()
            W_all = np.stack([W_T1, W_T2, W_T1GD, W_FLAIR], axis=-1)  ##!!  
            batch_X.append(W_all)
        batch_X = np.stack(batch_X, axis=0)  ##!!  
        return batch_X, batch_y


np.random.seed(random_seed)

#Instantiation
AlexNet = Sequential()

#1st Convolutional Layer
AlexNet.add(Conv3D(filters=64, input_shape=(240,240,155,4), kernel_size=5, strides=2, padding='same'))  ##!!  
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(MaxPooling3D(pool_size=3, strides=3, padding='same'))

#2nd Convolutional Layer
AlexNet.add(Conv3D(filters=128, kernel_size=3, strides=1, padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(MaxPooling3D(pool_size=3, strides=3, padding='same'))

#3rd Convolutional Layer
AlexNet.add(Conv3D(filters=192, kernel_size=3, strides=1, padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))

#4th Convolutional Layer
AlexNet.add(Conv3D(filters=192, kernel_size=3, strides=1, padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))

#5th Convolutional Layer
AlexNet.add(Conv3D(filters=128, kernel_size=3, strides=1, padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(MaxPooling3D(pool_size=3, strides=3, padding='same'))

#Passing it to a Fully Connected layer
AlexNet.add(Flatten())
# 1st Fully Connected Layer
AlexNet.add(Dense(256))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
# Add Dropout to prevent overfitting
AlexNet.add(Dropout(0.4))

#2nd Fully Connected Layer
AlexNet.add(Dense(256))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
#Add Dropout
AlexNet.add(Dropout(0.4))

#3rd Fully Connected Layer
AlexNet.add(Dense(64))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
#Add Dropout
AlexNet.add(Dropout(0.4))

#Output Layer
AlexNet.add(Dense(num_classes))  ##!!  
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('softmax'))

#Model Summary
AlexNet.summary()

AlexNet.compile(loss = keras.losses.categorical_crossentropy, optimizer= 'adam', metrics=['accuracy'])




#Learning Rate Annealer
from keras.callbacks import ReduceLROnPlateau
lrr= ReduceLROnPlateau(monitor='val_accuracy', factor=.01, patience=3, min_lr=1e-5) 

#Defining the parameters
batch_size = 4  ##!!  
epochs = 10  ##!!  
learn_rate = .001

#Training the model
AlexNet.fit(x=data_generator(x_train, y_train, batch_size=batch_size), epochs = epochs, steps_per_epoch = None, validation_data = data_generator(x_val, y_val, batch_size=batch_size), validation_steps = None, callbacks = [lrr], verbose=1, use_multiprocessing=True)  ##!!  

#After successful training, we will visualize its performance.

import matplotlib.pyplot as plt
#Plotting the training and validation loss

f,ax=plt.subplots(2,1) #Creates 2 subplots under 1 column

#Assigning the first subplot to graph training loss and validation loss
ax[0].plot(AlexNet.history.history['loss'],color='b',label='Training Loss')
ax[0].plot(AlexNet.history.history['val_loss'],color='r',label='Validation Loss')

#Plotting the training accuracy and validation accuracy
ax[1].plot(AlexNet.history.history['accuracy'],color='b',label='Training  Accuracy')
ax[1].plot(AlexNet.history.history['val_accuracy'],color='r',label='Validation Accuracy')

plt.legend()

#Defining function for confusion matrix plot
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels = np.arange(num_classes))  ##!!  
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#Print Confusion matrix
    fig, ax = plt.subplots(figsize=(7,7))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

np.set_printoptions(precision=2)

#Making prediction
y_predictions=AlexNet.predict(x=data_generator(x_set=x_test, y_set=y_test, batch_size=batch_size), use_multiprocessing=True)  ##!!  
y_pred=np.argmax(y_predictions,axis=1)  ##!!  
y_true=np.argmax(y_test,axis=1)

class_names = y_label  ##!!  

#Plotting the confusion matrix
from sklearn.metrics import confusion_matrix
confusion_mtx=confusion_matrix(y_true, y_pred, labels = np.arange(num_classes))  ##!!  

# Plotting non-normalized confusion matrix
plot_confusion_matrix(y_true, y_pred, classes = class_names,title = 'Confusion matrix, without normalization')

# Plotting normalized confusion matrix
plot_confusion_matrix(y_true, y_pred, classes=class_names, normalize=True, title='Normalized confusion matrix')

#Classification accuracy
from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_true, y_pred)
print('Accuracy Score = ', acc_score)

AlexNet.evaluate(x=data_generator(x_set=x_test, y_set=y_test, batch_size=batch_size), use_multiprocessing=True)  ##!!  

# %%
AlexNet.metrics_names


