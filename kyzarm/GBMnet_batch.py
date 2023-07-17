# GBMnet_batch.py
# Heavily derived from the following:
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence


from skimage.io import imread
from skimage.transform import resize
import numpy as np
import tensorflow as tf
import math

# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class GBMnet_batch(tf.keras.utils.Sequence):
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

        return np.array([
            resize(imread(file_name), (200, 200))
               for file_name in batch_x]), np.array(batch_y)