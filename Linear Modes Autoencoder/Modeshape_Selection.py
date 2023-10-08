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

#------------------------------------------------------------------------------------

Q_selection_list = [2]

#-----------------------------------------------------------------------------------

Q_string = []

file_path = r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\Linear Modes Autoencoder\Modal Response data\3DOF_Selected.text"

file = open(file_path, "w")

for j in Q_selection_list:
    for i in range(len(Q_data[:, 0])):

        Q_string.append(str(Q_data[:, j-1][i].numpy()))

    print(len(Q_string))

    Q_soln_string = ", ".join(Q_string)

    Q_string = []
    file.write(Q_soln_string + "\n")