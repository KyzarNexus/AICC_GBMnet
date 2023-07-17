# GBMnet_main.py
# Main script to run the CNN training and testing on the UPENN-GBM dataset.

# Main Tasks:
# 1. First iteration will be a replication of the ipynb notebook for 
# proof that the network works. 
# 2. Expand the original CNN to be approximate to the AlexNet CNN architecture. 
# 3. Create a rudimentary train/test dataset (nTr = 10, nTe = 10) that draws 
#   randomly from the dataset to test functionality. 
# 4. Import said dataset using the nibabel package. 
# 5. Test the expanded CNN on the trial train/test datasets to prove functionality. 
# 
# Supplimental Tasks:
# - Implement a source control for remote work on repo (Git or SVN, whatever everyone is most familiar with)
#   - Create initial commit and ensure that all members have access from their personal workstations. 
#   - Create a local branch on the SSH server that allows all members with access to pull/commit/push 
#       from that branch. 

# 1. Reimplement MNIST CNN in native python environment (COMPLETE)
#   - Current env uses Python 3.10.11
#   - NOTE: Uses the mnist sample dataset stored on the keras package. 
#       The prior 'mnist' package had errors with decompressing the dataset. 
# 2. Expand CNN to be approximate to the Aleksnet architecture. 



# Imports
import math
import numpy as np
from matplotlib import pyplot
from skimage.io import imread
from skimage.transform import resize
from keras.datasets import mnist
from keras import Model
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Conv3D
from keras.layers import MaxPooling3D
import tensorflow as tf
import GBMnet_batch # Batching class. 
from GBMnet_helpers import * # Helper functions

# Helper Functions/Classes. May need to be moved to a separate file later. 

# Q: What do x_set/y_set specify? The batch   image paths & labels?
# - To delete
class GBMnet_batch(tf.keras.utils.Sequence):
  # Here, `x_set` is list of path to the images
  # and `y_set` are the associated labels.
    def __init__(self, x_set, y_set, ImgX, ImgY, batch_size):
      # Should take in directory
      # Mode: Tr/Te/Val
        self.x, self.y = x_set, y_set
        self.ImgX, self.ImgY = ImgX, ImgY
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

        # Outputs an image np.array 
        return np.array([
            resize(imread(file_name), (ImgX, ImgY))
               for file_name in batch_x]), np.array(batch_y)



# Define the data loader
def data_loader(images, labels, batch_size):
    num_samples = images.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        yield images[batch_indices], labels[batch_indices]


def one_hot(label):
  new_label = np.zeros(10)
  new_label[label] = 1
  return new_label

def one_hot_array(labels):
  #Let's make an empty list to store our results.
  new_labels = []
  
  #We'll use a "for" loop to look at each label in the list.
  for label in labels:
    new_labels.append(one_hot(label))

  #Before we're done, we need to turn the list into a numpy array.  We almost always use numpy arrays with neural networks.
  return np.asarray(new_labels)


# Import
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Summary Information
print(train_images.shape)
print(test_images.shape)

print(train_labels.shape)
print(test_labels.shape)

indexes = list(range(60000))
i = np.random.choice(indexes)

pyplot.imshow(train_images[i])

print(train_labels[0:10])

### CNN Parameters
xImage = 28
yImage = 28
batchSize = 100

### Batching
# batchedData = GBMnet_batch()

### Construct CNN ###

# Input
image = Input(shape=(xImage, yImage, 1))
# Conv1
conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(image)
pool1 = MaxPooling2D()(conv1)
# Conv2
conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D()(conv2)
# Flatten
flat = Flatten()(pool2)
# Dense1
dense1 = Dense(100, activation='relu')(flat)
# Classifier
denseFinal = Dense(10, activation='softmax')(dense1)
model = Model(image,denseFinal)
# Summary
model.summary()


# Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
# model.fit(np.expand_dims(train_images, axis=-1), one_hot_array(train_labels))
# num_epochs = math.floor(train_images.shape[0]/batchSize)
num_epochs = 1
train_generator = data_loader(np.expand_dims(train_images, axis=-1), one_hot_array(train_labels), batch_size=batchSize)
for epoch in range(num_epochs):
    print("Epoch", epoch + 1)
    for batch_images, batch_labels in train_generator:
        model.train_on_batch(batch_images, batch_labels)


# Evaluate
model.evaluate(np.expand_dims(test_images, axis=-1), one_hot_array(test_labels))


