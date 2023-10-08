# General Math
import numpy as np
import math
import random
from scipy.integrate import odeint
from scipy import  interpolate
from time import time

# Tensorflow
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras.layers import BatchNormalization
from keras import activations
from tensorflow.python.ops.numpy_ops import np_config
from keras import regularizers
from tensorflow import keras
import tensorboard

# Plotting and Animations
import matplotlib.pyplot as plt
from matplotlib import animation,rc
from mpl_toolkits.mplot3d import Axes3D

# Miscellaneous
import sys
from time import time
from datetime import datetime
import os

Q_data_n = 3

Q_file_path = r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\Linear Modes Autoencoder\Modal Response data\3DOF_Original.text"

Q_data = []

# Open the text file and read its content
with open(Q_file_path, 'r') as Q_file:
    # Iterate through each line in the text file

    count = 0

    for line in Q_file:
        # Remove leading and trailing whitespace and then split by ", "

        Q_split_row = line.strip().split(", ")

        Q_split_row_float = []

        for i in Q_split_row:

            Q_split_row_float.append(float(i))
            
        # Append the split row to the matrix
        Q_data.append(Q_split_row_float)

Q_data = tf.transpose(tf.cast(tf.convert_to_tensor(Q_data), tf.float32))
Q_data = tf.reshape(Q_data, (1000, Q_data_n))

t = np.linspace(0, 100, 1000)

fig, (ax3) = plt.subplots(1,1,figsize=(12,6))

ax3.plot(t, Q_data[:, 0], color = "blue", label = "Original Modal Response 1")
ax3.plot(t, Q_data[:, 1], color = "red", label = "Original Modal Response 2")
ax3.plot(t, Q_data[:, 2], color = "orange", label = "Original Modal Response 3")
ax3.set(xlabel = "time(s)", ylabel = "x(m)")
ax3.grid()
ax3.legend(loc = "upper right")

plt.show()