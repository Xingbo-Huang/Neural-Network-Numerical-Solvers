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

# Initialize an empty matrix
x_data_tensor = []

# Specify the text file path
text_file_path = r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\Training data\Mass_Spring\Mass_Spring_Data_5DOF_Damped.txt"

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

    # x = FEM data
    # x dot = [1, 1, 1.....]
    # C is randomized with 2 coefficents
    # 2 basis functions used sin and cos

    #Load training data
    loaded_x_data_tensor = tf.cast(x_data_tensor, tf.float32)

    # Now, you can compute the intermediate outputs for 'x'
    H = model_H(tf.transpose(loaded_x_data_tensor))
    Q = model_Q(tf.transpose(loaded_x_data_tensor))

    W_1 = model_Encoder.get_layer('hidden_1').weights[0]

    loss_linear_uncorrelation = tf.reduce_max(tf.reduce_sum(tf.abs(tf_cov(H) - tf.eye(num_input))))
    loss_gaussianality = tf.reduce_max(tf.abs(1/4 * tf.pow(H, 4))) + tf.reduce_max(tf.reduce_sum(tf.abs(tf.matmul(W_1, tf.transpose(W_1)) - tf.eye(num_input))))
    loss_nonlinear_uncorrelation = tf.reduce_max(tf.reduce_sum(tf.abs(tf_cov(Q) - tf.eye(num_latent))))
    
    X_hat = model_Decoder(model_Encoder(tf.transpose(loaded_x_data_tensor)))

    loss_reconstruction = tf.reduce_mean(tf.reduce_sum(tf.abs(X_hat - tf.transpose(loaded_x_data_tensor)), axis = 0)) 

    loss_total = 5 * loss_linear_uncorrelation + loss_gaussianality + 5 * loss_nonlinear_uncorrelation + 0.4 * loss_reconstruction
    # loss_total = loss_reconstruction

    dummy1 = 0
    dummy2 = 0
    dummy3 = 0

    return loss_total, dummy1, dummy2, dummy3, Q, X_hat

def get_grad_CAE():
    
    g = 0

    with tf.GradientTape(persistent=True) as tape:
        # This tape is for derivatives with
        # respect to trainable variables

        loss, u_xa, ua, s, Q, X_hat = compute_loss_CAE()

        trainable_variables = model_Encoder.trainable_variables + model_Decoder.trainable_variables 

        tape.watch(trainable_variables)

    g = tape.gradient(loss, trainable_variables)
    del tape

    return loss, g, u_xa, ua, s, Q, X_hat

@tf.function
def train_step_CAE():

    #print("tran_step() is working")

    lossa, grad_theta_a, u_xa, ua, s, Q, X_hat= get_grad_CAE()

    trainable_variables = model_Encoder.trainable_variables + model_Decoder.trainable_variables

    optim_CAE.apply_gradients(zip(grad_theta_a, trainable_variables))

    return lossa, u_xa, ua, s, Q, X_hat

# Set data type
DTYPE = 'float32'

tf.keras.backend.set_floatx(DTYPE)

# Initialize model aka u_\theta

optim_CAE = tf.keras.optimizers.Adam(learning_rate=0.002)

#tf.keras.backend.clear_session()

num_input = len(x_data_tensor)
num_latent = len(x_data_tensor)
num_output = len(x_data_tensor)

model_Encoder = init_Encoder_model(num_input, num_latent, num_output)
model_Decoder = init_Decoder_model(num_input, num_latent, num_output)

# Create a new model using the desired layers as outputs
model_H = tf.keras.Model(inputs=model_Encoder.input, outputs=model_Encoder.get_layer("hidden_1").output)
model_Q = tf.keras.Model(inputs=model_Encoder.input, outputs=model_Encoder.get_layer("Encoder_output").output)
# model_physics = init_latent_model(10, 10, 2, 10)

predictions = 0

loss_list = []
epoch_list = []

model_Encoder.summary()
model_Decoder.summary()

k_n = 1
for k in range(k_n):

    num_epoch = 0

    # Number of training epochs for CAE
    M = 50000
    for i in range(M):
    
        num_epoch += 1

        #print("Reached train_step()")

        loss_CAE, u_x_CAE, u_CAE, s, Q, X_hat = train_step_CAE()
    
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

print(Q)

fig, (ax3) = plt.subplots(1,1,figsize=(12,6))

ax3.plot(t, Q[:, 0], color = "blue", label = "Modal Response 1")
ax3.plot(t, Q[:, 1], color = "red", label = "Modal Response 2")
ax3.plot(t, Q[:, 2], color = "orange", label = "Modal Response 3")
ax3.plot(t, Q[:, 3], color = "green", label = "Modal Response 4")
ax3.plot(t, Q[:, 4], color = "black", label = "Modal Response 5")
ax3.set(xlabel = "time(s)", ylabel = "q(m)")
ax3.grid()
ax3.legend(loc = "upper right")

plt.show()

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))

prediction = []

W = model_Decoder.get_layer('output').weights[0]

print(W)

x = np.linspace(0, 1, 5)

Q_string = []

file_path = r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\Linear Modes Autoencoder\Modal Response data\5DOF_Damped_Original.text"

file = open(file_path, "w")

for j in range(5):
    for i in range(len(Q[:, 0])):

        Q_string.append(str(Q[:, j][i].numpy()))

    print(len(Q_string))

    Q_soln_string = ", ".join(Q_string)

    Q_string = []
    file.write(Q_soln_string + "\n")

#Autoencoder error very small compared to PDE error, output and input almost overlapping.
ax1.plot(x, W[0], label = 'Modeshape 1', color='green', marker = "o")
ax1.plot(x, W[1], label = 'Modeshape 2', color='Orange', marker = "o")
ax1.plot(x, W[2], label = 'Modeshape 3', color='blue', marker = "o")
ax1.plot(x, W[3], label = 'Modeshape 4', color='red', marker = "o")
ax1.plot(x, W[4], label = 'Modeshape 5', color='black', marker = "o")

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

fig, (ax5, ax6, ax7) = plt.subplots(1,3,figsize=(18,6))

ax5.plot(t, x_data_tensor[0, :], color = "blue", label = "X1")
ax5.plot(t, X_hat[:, 0], color = "green", label = "X hat 1")
ax6.plot(t, x_data_tensor[1, :], color = "blue", label = "X2")
ax6.plot(t, X_hat[:, 1], color = "green", label = "X hat 2")
ax7.plot(t, x_data_tensor[1, :], color = "blue", label = "X3")
ax7.plot(t, X_hat[:, 1], color = "green", label = "X hat 3")

ax5.set(xlabel = "time(s)", ylabel = "x(m)")
ax5.grid()
ax5.legend(loc = "upper right")

ax6.set(xlabel = "time(s)", ylabel = "x(m)")
ax6.grid()
ax6.legend(loc = "upper right")

ax7.set(xlabel = "time(s)", ylabel = "x(m)")
ax7.grid()
ax7.legend(loc = "upper right")

plt.show()