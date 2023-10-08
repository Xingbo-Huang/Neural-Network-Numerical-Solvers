#inear modes autoencoder

#Started: 2023/07/24
#Author: Xingbo Huang

#----------------------------------------------------------------------------------

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

Q_data_n = 1

# Initialize an empty matrix
x_data_tensor = []

# Specify the text file path
text_file_path = r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\Training data\Mass_Spring\Mass_Spring_Data_3DOF.txt"

# Open the text file and read its content
with open(text_file_path, 'r') as text_file:
    # Iterate through each line in the text file

    count = 0

    for line in text_file:
        # Remove leading and trailing whitespace and then split by ", "

        split_row = line.strip().split(", ")

        split_row_float = []

        for i in split_row:

            split_row_float.append(float(i))
            
        # Append the split row to the matrix
        x_data_tensor.append(split_row_float)

x_data_tensor = tf.cast(tf.convert_to_tensor(x_data_tensor), tf.float32)

#-----------------------------------------------------------------------------------------------------

Q_file_path = r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\Linear Modes Autoencoder\Modal Response data\3DOF_Selected.text"

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

def tf_cov(x):
    # Step 1: Calculate the means of each column (variable)
    means = tf.reduce_mean(x, axis=0)
    
    # Step 2: Center the data by subtracting the means from each element
    centered_data = x - means
    
    # Step 3: Calculate the covariance matrix
    n = tf.shape(x)[0]
    covariance_matrix = tf.matmul(centered_data, centered_data, transpose_a=True) / tf.cast(n - 1, tf.float32)
    
    return covariance_matrix

def init_Encoder_model(input_dim, latent_dim, output_dim):

    # Initialize a feedforward neural network
    model = tf.keras.Sequential()

    # Input layer
    model.add(tf.keras.Input((input_dim)))

    model.add(tf.keras.layers.Dense(input_dim, name = "hidden_1", use_bias= False))

    # Output layer
    model.add(tf.keras.layers.Dense(latent_dim, activation=tf.keras.activations.get('tanh'), name = "Encoder_output"                               
    ,use_bias= False
    # ,kernel_regularizer= regularizers.L2(),
    # bias_regularizer= regularizers.L2()
    ))

    return model

def init_Decoder_model(input_dim, latent_dim, output_dim):
    # Initialize a feedforward neural network

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input((latent_dim)))

    # Output layer
    model.add(tf.keras.layers.Dense((output_dim), name = "output", use_bias= False))

    return model
    
def compute_loss_CAE():

    loaded_x_data_tensor = tf.cast(x_data_tensor, tf.float32)

    Q = model_Q(tf.transpose(loaded_x_data_tensor))

    X_hat = model_Decoder(model_Encoder(tf.transpose(loaded_x_data_tensor)))

    loss_Q_match = 0

    for i in range(Q_data_n):

        loss_Q_match += tf.reduce_mean(tf.square(Q[:, i] - Q_data[:, i]))

    loss_reconstruction = tf.reduce_mean(tf.square(X_hat - tf.transpose(loaded_x_data_tensor))) 

    loss_total = loss_reconstruction + loss_Q_match

    dummy1 = 0
    dummy2 = 0
    dummy3 = 0

    return loss_total, dummy1, dummy2, dummy3, Q

def get_grad_CAE():
    
    g = 0

    with tf.GradientTape(persistent=True) as tape:
        # This tape is for derivatives with
        # respect to trainable variables

        loss, u_xa, ua, s, Q = compute_loss_CAE()

        trainable_variables = model_Encoder.trainable_variables + model_Decoder.trainable_variables 

        tape.watch(trainable_variables)

    g = tape.gradient(loss, trainable_variables)
    del tape

    return loss, g, u_xa, ua, s, Q

@tf.function
def train_step_CAE():

    #print("tran_step() is working")

    lossa, grad_theta_a, u_xa, ua, s, Q = get_grad_CAE()

    trainable_variables = model_Encoder.trainable_variables + model_Decoder.trainable_variables

    optim_CAE.apply_gradients(zip(grad_theta_a, trainable_variables))

    return lossa, u_xa, ua, s, Q

# Set data type
DTYPE = 'float32'

tf.keras.backend.set_floatx(DTYPE)

# Initialize model aka u_\theta

optim_CAE = tf.keras.optimizers.Adam(learning_rate=0.0005)

#tf.keras.backend.clear_session()

num_input = 3
num_latent = 1
num_output = 3

model_Encoder = init_Encoder_model(num_input, num_latent, num_output)
model_Decoder = init_Decoder_model(num_input, num_latent, num_output)

# Create a new model using the desired layers as outputs
model_H = tf.keras.Model(inputs=model_Encoder.input, outputs=model_Encoder.get_layer("hidden_1").output)
model_Q = tf.keras.Model(inputs=model_Encoder.input, outputs=model_Encoder.get_layer("Encoder_output").output)
# model_physics = init_latent_model(10, 10, 2, 10)

predictions = 0

loss_list = []
epoch_list = []

k_n = 1
for k in range(k_n):

    num_epoch = 0

    # Number of training epochs for CAE
    M = 10000
    for i in range(M):
    
        num_epoch += 1

        #print("Reached train_step()")

        loss_CAE, u_x_CAE, u_CAE, s, Q = train_step_CAE()
    
        if num_epoch%5 == 0 :
        
            print("Epoch: ", num_epoch)

        epoch_list.append(num_epoch)
        loss_list.append(loss_CAE)

        #lossb = train_step_b()

    #print("pts avg", np.average(pts))
    #print("pts length", len(pts))
    #print("j:", j)
        
    #print("pred a", predictions_a[half - 1])
    #print("pred b", predictions_b[half - 1])

    #print("exact", exact_2[half -1])

#for i in predictions:

    #prediction.append(i)

# x_plot = np.linspace(0, 1, len(x_data))
# x_latent_plot = np.linspace(0, 1, num_latent)

# PINN_x_plot = np.linspace(0, 1, num_of_training_pts)

t = np.linspace(0, 100, 1000)

fig, (ax4) = plt.subplots(1,1,figsize=(12,6))

ax4.plot(t, Q_data[:, 0], color = "blue", label = "Original Modal Response 1")
# ax4.plot(t, Q_data[:, 1], color = "red", label = "Original Modal Response 2")
ax4.set(xlabel = "time(s)", ylabel = "x(m)")
ax4.grid()
ax4.legend(loc = "upper right")

fig, (ax3) = plt.subplots(1,1,figsize=(12,6))

ax3.plot(t, Q[:, 0], color = "blue", label = "New Modal Response 1")
# ax3.plot(t, Q[:, 1], color = "red", label = "New Modal Response 2")
ax3.set(xlabel = "time(s)", ylabel = "x(m)")
ax3.grid()
ax3.legend(loc = "upper right")

plt.show()

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))

prediction = []

W = model_Decoder.get_layer('output').get_weights()[0]

print("weights:", W)

x = np.linspace(0, 1, 3)

model_Decoder.summary()

#Autoencoder error very small compared to PDE error, output and input almost overlapping.
ax1.plot(x, tf.squeeze(W[0]), label = 'Linear Mode 1', color='green', marker = "o")
# ax1.plot(x, tf.squeeze(W[1]), label = 'Linear Mode 2', color='green', marker = "o")

#ax1.plot(x_plot, FDM_data, label = '25 node FDM_Solution', color='pink')
#ax1.plot(PINN_x_plot, PINN_representation, label = 'PINN_Solution', color='yellow')
#ax1.plot(points[b_pts:],predictions_b, label = 'Approx Full', color='green')

#exact_1 = (points[:a_pts]**4/24)
#exact_2 = (points[b_pts:]**2/8 - 1/24)
#exact   = np.append(exact_1,exact_2)

#exact = -points*points*points/6 + 3*points/2

#Continous bar true solution
# exact_1 = -points[:a_pts]*points[:a_pts]*points[:a_pts]/6 + 3*points[:a_pts]/2
# exact_2 = -points[b_pts:]*points[b_pts:]*points[b_pts:]/6 + 3*points[b_pts:]/2
# exact = np.hstack((exact_1, exact_2))

# x_true = np.linspace(0, 1, num_of_training_pts)
# e = np.e *np.ones(len(points))
# exact = -e/(-e**2-1) * e**x_true + e/(-e**2-1)*e**(-x_true)

#exact_1 = -points[:a_pts]*points[:a_pts]*points[:a_pts]/6 + 8*points[:a_pts]/8
#exact_2 = -points[b_pts:]*points[b_pts:]*points[b_pts:]/3 + 2*points[b_pts:] - 23/48
# exact   = np.append(exact_1,exact_2)

# residual = 0

# for i in range(len(autoencoder_output_vector) - 1):

#     residual += 1/len(autoencoder_output_vector) * (abs(autoencoder_output_vector[i+1] - FEM_data[i+1])/abs(FEM_data[i+1]))

# print("Residual:", residual)

#exact_1_der = -pts[:half+1]**2/2 + 2
#exact_2_der = -pts[half+1:]**2 + 2
#exact_der   = np.append(exact_1_der,exact_2_der)

#exact_1_sec = -pts[:half+1]
#exact_2_sec = -(1/A)*pts[half+1:]
#exact_sec   = np.append(exact_1_sec,exact_2_sec)
# ax1.plot(points, exact, label = "Exact solution", color='purple')

#ax1.plot(epoch_list, loss_list, color = "r")

#ax1.plot(points[:a_pts],u_xa, '--',label = "approx derivative", color='blue')
#ax1.plot(points[b_pts - 1:],u_xb, '--',label = "approx derivative", color='blue')

#ax1.plot(pts,exact_der, label = "exact derivative", color='blue')
#ax1.plot(points[:a_pts],u_xxa, '--',label = "approx second der", color='red')
#ax1.plot(points[b_pts - 1:],u_xxb, '--',label = "approx second der", color='red')

#ax1.plot(pts,exact_sec, label = "exact second der", color='red')

ax1.grid()
ax1.legend()
ax2.plot(epoch_list, loss_list, label = "Loss Histroy", color = "red")
ax2.grid()
ax2.legend()
#ax2.plot([(i+1)*10 for i in range(len(losses1))],losses1, label = "Trained Whole", color='blue')

plt.show()

#ax2.set_yscale('log')
#ax2.legend()