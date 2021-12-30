
__author__ = 'Majd Jamal'

import numpy as np
from utils.params import params
from tensorflow.data import Dataset
import tensorflow as tf

def getData():
    #Load data as shape = (Npts, 256, 256, 3). X as grey and y as color
    data_grey = np.load('data/processed_data/X_train.npy') / 255
    data_color = np.load('data/processed_data/y_train.npy') / 255

    data_grey = tf.cast(data_grey, tf.float32)
    data_color = tf.cast(data_color, tf.float32)

    train_dataset = Dataset.from_tensor_slices((data_grey, data_color))

    return train_dataset
