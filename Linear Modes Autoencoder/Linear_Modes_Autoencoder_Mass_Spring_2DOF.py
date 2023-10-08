#inear modes autoencoder for 2DOF system

#Author: Xingbo Huang

#----------------------------------------------------------------------------------

# General Math Libraries
import numpy as np
import math
import random
from scipy.integrate import odeint
from scipy import  interpolate
from time import time

# Tensorflow Libraries
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras.layers import BatchNormalization
from keras import activations
from tensorflow.python.ops.numpy_ops import np_config
from keras import regularizers
from tensorflow import keras
import tensorboard

# Plotting 2D and 3D
import matplotlib.pyplot as plt
from matplotlib import animation,rc
from mpl_toolkits.mplot3d import Axes3D

# Miscellaneous
import sys
from time import time
from datetime import datetime
import os

# Initialize imported tensor
x_data_tensor = []

# Specify the text file path
text_file_path = r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\Training data\Mass_Spring\Mass_Spring_Data.txt"

# Open the text file and read its content
with open(text_file_path, 'r') as text_file:

    # Iterate through each line in the text file
    count = 0

    for line in text_file:

        # RSplit by ", " in the csv file to extract contents
        split_row = line.strip().split(", ")

        split_row_float = []

        for i in split_row:

            split_row_float.append(float(i))
            
        # Append the split row to the matrix
        x_data_tensor.append(split_row_float)

x_data_tensor = tf.cast(tf.convert_to_tensor(x_data_tensor), tf.float32)

print(x_data_tensor)

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
    # Bias and kernel regularization can be turned on if needed.
    model = tf.keras.Sequential()

    # Input layer
    model.add(tf.keras.Input((input_dim)))

    # Hidden layer (activation: linear)
    model.add(tf.keras.layers.Dense(input_dim, name = "hidden_1", use_bias= False))

    # Output layer (activation: tanh)
    model.add(tf.keras.layers.Dense(latent_dim, activation=tf.keras.activations.get('tanh'), name = "Encoder_output"                               
    ,use_bias= False
    # ,kernel_regularizer= regularizers.L2(),
    # bias_regularizer= regularizers.L2()
    ))

    return model

def init_Decoder_model(input_dim, latent_dim, output_dim):

    # Initialize a feedforward neural network

    #Input layer (also output of Encoder)
    model = tf.keras.Sequential()

    #Weights between input and output of the decoder are the linear modes

    # Output layer
    model.add(tf.keras.layers.Dense((output_dim), name = "output", use_bias= False))

    return model

#Tensorflow loss definition
def compute_loss_Autoencoder():

    # Time series data of 2DOF spring mass system
    loaded_x_data_tensor = tf.cast(x_data_tensor, tf.float32)

    # H is the hidden layer output of the Encoder
    # Q is the output layer output of the Encoder
    H = model_H(tf.transpose(loaded_x_data_tensor))
    Q = model_Q(tf.transpose(loaded_x_data_tensor))

    # Weights between the input and output layers of the Decoder (Linear modes)
    W_1 = model_Encoder.get_layer('hidden_1').weights[0]

    # Linear uncorrelation loss term
    loss_linear_uncorrelation = tf.reduce_max(tf.reduce_sum(tf.abs(tf_cov(H) - tf.eye(num_input))))

    # Non-gaussianality loss term
    loss_gaussianality = tf.reduce_max(tf.abs(1/4 * tf.pow(H, 4))) + tf.reduce_max(tf.reduce_sum(tf.abs(tf.matmul(W_1, tf.transpose(W_1)) - tf.eye(num_input))))

    # Nonlinear uncorrelation loss term 
    loss_nonlinear_uncorrelation = tf.reduce_max(tf.reduce_sum(tf.abs(tf_cov(Q) - tf.eye(num_latent))))
    
    # X hat is the output of the decoder
    X_hat = model_Decoder(model_Encoder(tf.transpose(loaded_x_data_tensor)))

    # Reconstruction loss term
    loss_reconstruction = tf.reduce_mean(tf.reduce_sum(tf.abs(X_hat - tf.transpose(loaded_x_data_tensor)), axis = 0)) 

    # Total loss term with loss weights
    loss_total = 5 * loss_linear_uncorrelation + loss_gaussianality + 5 * loss_nonlinear_uncorrelation + 0.3 * loss_reconstruction
    # loss_total = loss_reconstruction

    #Dummy variables please ignore
    dummy1 = 0
    dummy2 = 0
    dummy3 = 0

    return loss_total, dummy1, dummy2, dummy3, Q, X_hat

#Tensorflow gradient computation
def get_grad_Autoencoder():
    
    g = 0

    #Gradient tape
    with tf.GradientTape(persistent=True) as tape:

        loss, u_xa, ua, s, Q, X_hat = compute_loss_Autoencoder()

        trainable_variables = model_Encoder.trainable_variables + model_Decoder.trainable_variables 

        tape.watch(trainable_variables)

    #Obtain gradient of loss with respect to theta
    g = tape.gradient(loss, trainable_variables)
    del tape

    return loss, g, u_xa, ua, s, Q, X_hat

#Gradient descent, apply graph execution
@tf.function
def train_step_Autoencoder():

    #print("tran_step() is working")

    lossa, grad_theta_a, u_xa, ua, s, Q, X_hat= get_grad_Autoencoder()

    trainable_variables = model_Encoder.trainable_variables + model_Decoder.trainable_variables

    optim_CAE.apply_gradients(zip(grad_theta_a, trainable_variables))

    return lossa, u_xa, ua, s, Q, X_hat

# Set data type
DTYPE = 'float32'

tf.keras.backend.set_floatx(DTYPE)

# Initialize model with alpha = learning late
optim_CAE = tf.keras.optimizers.Adam(learning_rate=0.01)

#Neural network architecture hyperparamters
num_input = len(x_data_tensor)
num_latent = len(x_data_tensor)
num_output = len(x_data_tensor)

# Initialize models
model_Encoder = init_Encoder_model(num_input, num_latent, num_output)
model_Decoder = init_Decoder_model(num_input, num_latent, num_output)

# Create a new model using the desired layers as outputs
model_H = tf.keras.Model(inputs=model_Encoder.input, outputs=model_Encoder.get_layer("hidden_1").output)
model_Q = tf.keras.Model(inputs=model_Encoder.input, outputs=model_Encoder.get_layer("Encoder_output").output)

#Initialize loss vs epoch graph
loss_list = []
epoch_list = []

starttime = time()

#Neural network training using max epoch loop
k_n = 1
for k in range(k_n):

    num_epoch = 0

    # Number of training epochs for CAE
    M = 5000
    for i in range(M):
    
        num_epoch += 1

        #print("Reached train_step()")

        loss_CAE, u_x_CAE, u_CAE, s, Q, X_hat = train_step_Autoencoder()
    
        if num_epoch%5 == 0 :
        
            print("Epoch: ", num_epoch)

        epoch_list.append(num_epoch)
        loss_list.append(loss_CAE)

#Run time
endtime = time()
runtime = endtime - starttime

#Below are all plots
print("Alorithm runtime:",runtime)

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))

prediction = []

W = model_Decoder.get_layer('output').weights[0]

print(W)

x = np.linspace(0, 1, 2)

Q_string = []
for i in range(len(Q[:, 0])):

    Q_string.append(str(Q[:, 0][i].numpy()))

Q_soln_string = ",".join(Q_string)

file_path = r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\Linear Modes Autoencoder\Modal Response data\2DOF.text"

file = open(file_path, "w")
file.write(Q_soln_string)


ax1.plot(x, W[0], label = 'Linear Mode 1', color='green', marker = "o")
ax1.plot(x, W[1], label = 'Linear Mode 2', color='Orange', marker = "o")

ax1.grid()
ax1.legend()
ax2.plot(epoch_list, loss_list, label = "Loss Histroy", color = "red")
ax2.grid()
ax2.legend()

plt.show()

t = np.linspace(0, 10, 1000)

print(Q)

fig, (ax3) = plt.subplots(1,1,figsize=(12,6))

ax3.plot(t, Q[:, 0], color = "blue", label = "Modal Response 1")
ax3.plot(t, Q[:, 1], color = "red", label = "Modal Response 2")
ax3.set(xlabel = "time(s)", ylabel = "q(m)")
ax3.grid()
ax3.legend(loc = "upper right")

plt.show()

fig, (ax5, ax6) = plt.subplots(1,2,figsize=(12,6))

ax5.plot(t, x_data_tensor[0, :], color = "blue", label = "X1")
ax5.plot(t, X_hat[:, 0], color = "green", label = "X hat 1")
ax6.plot(t, x_data_tensor[1, :], color = "blue", label = "X2")
ax6.plot(t, X_hat[:, 1], color = "green", label = "X hat 2")

ax5.set(xlabel = "time(s)", ylabel = "x(m)")
ax5.grid()
ax5.legend(loc = "upper right")

ax6.set(xlabel = "time(s)", ylabel = "x(m)")
ax6.grid()
ax6.legend(loc = "upper right")

plt.show()