#Optimize entire matrix equivalent to optimizing each matrix element
#Working version

#Using jacobian instead of gradient tape

#Started: 2023/07/04
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

#Import files

# from Non_Linear_Pendulum_Data import get_pendulum_data

#Import x data

file_path = r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\Training data\Dynamical Systems\x data"

file = open(file_path, "r")

x_data_string = file.read()
count_list = x_data_string.split(",")

x_data = []
for i in range(len(count_list)):

    x_data.append(float(count_list[i]))

#Import training data from FEM solver------------------------------------------------------------------

file_path = r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\Training data\Autonomous ODE FEM data.txt"

file = open(file_path, "r")

FEM_data_string = file.read()
count_list = FEM_data_string.split(",")

FEM_data = []
for i in range(len(count_list)):

    FEM_data.append(float(count_list[i]))

#Import training data from FDM solver------------------------------------------------------------------

file_path = r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\Training data\FDM Displacement data.txt"

file = open(file_path, "r")

FDM_data_string = file.read()
count_list = FDM_data_string.split(",")

FDM_data = []
for i in range(len(count_list)):

    FDM_data.append(float(count_list[i]))

#Import Strong Form PINN solution-----------------------------------

# Load the saved model
PINN_model = tf.keras.models.load_model(r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\NN_Models\1NN_Strong_PINN_model")

PINN_representation = tf.keras.Model(inputs=PINN_model.input, outputs=PINN_model.get_layer("output").output)

num_of_training_pts = 400
# Provide an input sample
input_sample = points = np.linspace(0, 1, num_of_training_pts) # Your input data
input_tensor = tf.cast(tf.convert_to_tensor(input_sample), tf.float32)

# Get the latent space representation
PINN_representation = PINN_representation.predict(input_tensor)

#-------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------


#CAE Neural network functions--------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

def init_Encoder_model(input_dim, latent_dim, output_dim):
    # Initialize a feedforward neural network

    model = tf.keras.Sequential()

    # Input layer
    model.add(tf.keras.Input(input_dim))

    model.add(tf.keras.layers.Dense(input_dim//2, activation=tf.keras.activations.get('sigmoid'), name = "hidden_1"
    ,kernel_initializer=tf.initializers.GlorotUniform()                                
    ,bias_initializer='zeros'
    # ,kernel_regularizer= regularizers.L2(),
    # bias_regularizer= regularizers.L2()
    ))

    # Latent layer
    model.add(tf.keras.layers.Dense(latent_dim, activation=tf.keras.activations.get('sigmoid'), name = "Encoder_output"
    ,kernel_initializer=tf.initializers.GlorotUniform()    
    ,bias_initializer='zeros'
    # ,kernel_regularizer= regularizers.L2(),
    # bias_regularizer= regularizers.L2()
    ))

    return model

def init_Decoder_model(input_dim, latent_dim, output_dim):
    # Initialize a feedforward neural network

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(latent_dim, activation=tf.keras.activations.get('sigmoid'), name = "Decoder_Input"
    ,kernel_initializer=tf.initializers.GlorotUniform()    
    ,bias_initializer='zeros'
    # ,kernel_regularizer= regularizers.L2(),
    # bias_regularizer= regularizers.L2(),
    ))

    model.add(tf.keras.layers.Dense(output_dim//2, activation=tf.keras.activations.get('sigmoid'), name = "hidden_2"
    ,kernel_initializer=tf.initializers.GlorotUniform()    
    ,bias_initializer='zeros'
    # ,kernel_regularizer= regularizers.L2(),
    # bias_regularizer= regularizers.L2()
    ))

    # Output layer
    model.add(tf.keras.layers.Dense(output_dim, name = "output"))

    return model

def get_r_CAE(model_encoder, model_decoder, X_r):
    
    # A tf.GradientTape is used to compute derivatives in TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        # Split t and x to compute partial derivatives
        x = X_r

        # Variables t and x are watched during tape
        # to compute derivatives u_t and u_x
        input = tf.reshape(x, (1, len(x_data)))

        tape.watch(input)
        z = model_Encoder(input)
        # print("z", z)

        y = model_Decoder(z)
        # Compute gradient u_x within the GradientTape
        # since we need second derivatives

    dydz = tape.jacobian(y, z)

    dzdx = tape.jacobian(z, input)
    # dydx = tape.jacobian(y, input)
       
    del tape

    x_hat = y

    return (dzdx, dydz, x_hat, x, z)
    
def compute_loss_CAE():

    # x = FEM data
    # x dot = [1, 1, 1.....]
    # C is randomized with 2 coefficents
    # 2 basis functions used sin and cos

    #Load training data
    x_data_tensor = tf.reshape(tf.cast(tf.convert_to_tensor(x_data), tf.float32), (len(x_data), 1))

    #load dxdt data
    dxdt_file = []

    dxdt = tf.reshape(load_dxdt(dxdt_file), (len(x_data), 1))
    
    #Obtain gradients
    dzdx, dydz, x_hat, x, z = get_r_CAE(model_Encoder, model_Decoder, x_data_tensor)
    z = tf.reshape(z, (num_latent, 1))
    x_hat = tf.reshape(x_hat, (len(x_data), 1))
    # print("dzdx", dzdx)
    # print("dxdt 1", dxdt_1)
    # print("dxdt 2", dxdt_2)
    # print("dydz", dydz)
    #SINDy dzdt
    theta_z = tf.cast(tf.convert_to_tensor(SINDy_theta_z_Representation(binary, initial_coe, z)), tf.float32)
    SINDy_coefficent_martrix = tf.cast(SINDy_Coefficents(initial_coe), tf.float32)

    # print("Sindy coefficent martrix:", SINDy_coefficent_martrix)

    dzdt_SINDy = tf.matmul(theta_z, SINDy_coefficent_martrix)

    # print("first", tf.matmul(dzdx, dxdt_2))
    # print("second", tf.matmul(dydz, dzdt_SINDy))

    loss_x_dot = tf.reduce_mean((dxdt - tf.matmul(dydz, dzdt_SINDy)) ** 2)
    loss_z_dot = tf.reduce_mean((tf.matmul(dzdx, dxdt) - dzdt_SINDy) ** 2)

    print("zdot NN",tf.matmul(dzdx, dxdt))
    print("z", z)
    print("SINDy coe:", SINDy_coefficent_martrix)
    print("zdot Sindy", dzdt_SINDy)

    loss_reconstruction = tf.cast(tf.reduce_mean((x_hat - x_data_tensor) ** 2), tf.float32)

    loss_regularization = tf.cast(tf.reduce_sum(abs(initial_coe)), tf.float32)

    # loss_total = loss_reconstruction 
    loss_total = 10 * loss_reconstruction + loss_x_dot + loss_z_dot + 5E-6 * loss_regularization
    # loss_total = (2 * loss_reconstruction + loss_x_dot + loss_z_dot + 0.01 * loss_regularization)
    # loss_total = (loss_reconstruction + 0.5 * loss_z_dot)
    
    return loss_total, dzdx, x_hat, SINDy_coefficent_martrix

def get_grad_CAE():
    
    g = 0

    with tf.GradientTape(persistent=True) as tape:
        # This tape is for derivatives with
        # respect to trainable variables

        loss, u_xa, ua, s = compute_loss_CAE()

        trainable_variables = model_Encoder.trainable_variables + model_Decoder.trainable_variables 

        trainable_variables.append(initial_coe)

        tape.watch(trainable_variables)

    g = tape.gradient(loss, trainable_variables)
    del tape

    return loss, g, u_xa, ua, s

@tf.function
def train_step_CAE():

    #print("tran_step() is working")

    lossa, grad_theta_a, u_xa, ua, s = get_grad_CAE()

    trainable_variables = model_Encoder.trainable_variables + model_Decoder.trainable_variables

    trainable_variables.append(initial_coe)

    optim_CAE.apply_gradients(zip(grad_theta_a, trainable_variables))

    return lossa, u_xa, ua, s

#Latent layer NN
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
'''
    List of functions inside the sindy library

    [1,
     z,
     z ** 2,
     z ** 3,
     sin(z),
     cos(z),
    ]
    
'''

def one(z):

    return tf.pow(z, 0)

def z(z):

    return tf.pow(z, 1)

def z_squared(z):

    return tf.pow(z, 2)

def z_cubed(z):

    return tf.pow(z, 3)

def sin_z(z):

    return tf.sin(z)

def cos_z(z):

    return tf.cos(z)

def x (x):

    return x
#----------------------------------------------------------------------------------

def SINDy_Coefficents(initial_coe):

    SINDy_coeff_matrix = initial_coe

    return SINDy_coeff_matrix

def SINDy_Library_Generation(list1, list2):

    SINDy_coeff_matrix = SINDy_Coefficents(list2)

    full_function_library = [one,
                             z, 
                             tf.square,
                             z_cubed,
                             tf.sin,
                             tf.cos
                            ]

    used_function_library = []

    indices_of_ones = [index for index, value in enumerate(list1) if value == 1]
    print(indices_of_ones)

    for i in indices_of_ones:

        used_function_library.append(full_function_library[i])

    return used_function_library, full_function_library

def apply_function(used_function_library, z):

    applied_function_list = tf.ones((num_latent, 1))
    # applied_function_list = tf.ones((test_z_dim, 1))

    for i in range(len(used_function_library)):

        # if i == 0: 

        #     print("if loop")

        #     print(used_function_library[i](5))
        #     function_computation = used_function_library[i]
        #     function_computation = np.reshape(function_computation, (-1, 1))
        
        function_computation = used_function_library[i](z)
        function_computation = tf.reshape(function_computation, (-1, 1))
        print("function computation:", function_computation)

        applied_function_list = tf.concat((applied_function_list, function_computation), axis = 1)
        print("Applied function list:", applied_function_list)

    #Remove junk items
    applied_function_list = applied_function_list[:, 1:]
    # print("theta of z", applied_function_list)

    applied_function_list_transpose = tf.transpose(applied_function_list)
    #print(applied_function_list_transpose)

    return applied_function_list_transpose, applied_function_list

def SINDy_theta_z_Representation(list1, list2, z):

    SINDy_coeff_matrix = SINDy_Coefficents(list2)
    used_function_library, full_function_library = SINDy_Library_Generation(list1, list2)

    theta_z_transpose, theta_z = apply_function(used_function_library, z)

    return theta_z

def load_dxdt(file):

    file_path = r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\Training data\Dynamical Systems\ x dot data"

    file = open(file_path, "r")

    dxdt_data_string = file.read()
    count_list = dxdt_data_string.split(",")

    dxdt_data = []
    for i in range(len(count_list)):

        dxdt_data.append(float(count_list[i]))

    return dxdt_data


#Running the neural network----------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
# Set data type
DTYPE = 'float32'

tf.random.set_seed(42)

tf.keras.backend.set_floatx(DTYPE)

# Initialize model aka u_\theta

optim_CAE = tf.keras.optimizers.Adam(learning_rate=0.01)

#tf.keras.backend.clear_session()

model_CAE = 0

num_input = len(x_data)
num_latent = 10
num_output = len(x_data)

model_Encoder = init_Encoder_model(num_input, num_latent, num_output)
model_Decoder = init_Decoder_model(num_input, num_latent, num_output)
# model_physics = init_latent_model(10, 10, 2, 10)

#model_b.summary()
#print("weights:", model_b.layers[1].get_weights()[1])

# [1, z, z^2, z^3, sinz, cosz]
binary = [0, 0, 1, 0, 1, 1]

num_basis_function = binary.count(1)

# count = 0

# for i in binary:
    
#     if i == 1:

#         count += 1

count = 1

#Key point to move this to initialize in the beginning as tf.Variable()
#initial_coe = tf.Variable(np.random.uniform(low=0, high=1, size = (num_basis_function, count)))
# initial_coe = tf.Variable(np.reshape(np.random.rand(num_basis_function, 1), (num_basis_function, 1)))
initial_coe = tf.Variable(np.reshape(np.array([0.3, -0.6, 0.3]), (num_basis_function, 1)))

# test_z_dim = 2
# z_list = tf.cast(tf.convert_to_tensor(np.array([1, 2])), tf.float32)

# print("Final result:", SINDy_theta_z_Representation(binary, initial_coe, z_list))

# sys.exit()

predictions = 0

loss_list = []
epoch_list = []

k_n = 1
for k in range(k_n):

    num_epoch = 0

    # Number of training epochs for CAE
    M = 2000
    for i in range(M):
    
        num_epoch += 1

        #print("Reached train_step()")

        loss_CAE, u_x_CAE, u_CAE, s = train_step_CAE()
    
        if num_epoch%5 == 0 :
        
            print("Epoch: ", num_epoch)

        if num_epoch//100 == 0 or num_epoch == M - 1:

            print("SINDy coefficent: ", s)

        epoch_list.append(num_epoch)
        loss_list.append(loss_CAE)

        #lossb = train_step_b()

    #print("pts avg", np.average(pts))
    #print("pts length", len(pts))
    #print("j:", j)
        
    #print("pred a", predictions_a[half - 1])
    #print("pred b", predictions_b[half - 1])

    #print("exact", exact_2[half -1])

model_Encoder.save(r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\NN_Models\Autoencoder_model\SINDy_Encoder")
model_Decoder.save(r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\NN_Models\Autoencoder_model\SINDy_Decoder")

#Latent Space training-------------------------------------------------------------------

Encoder = tf.keras.models.load_model(r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\NN_Models\Autoencoder_model\SINDy_Encoder")
Decoder = tf.keras.models.load_model(r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\NN_Models\Autoencoder_model\SINDy_Decoder")

# Provide an input sample
input_sample = x_data  # Your input data
input_tensor = tf.cast(tf.convert_to_tensor(input_sample), tf.float32)
test_input = tf.reshape(input_tensor, (1, len(x_data)))

# Get the latent space representation
latent_representation = Encoder.predict(test_input)
latent_representation_vector = tf.reshape(latent_representation, (-1))
autoencoder_output = Decoder.predict(Encoder.predict(test_input))
autoencoder_output_vector = tf.reshape(autoencoder_output, (-1))

# print("Original FEM Displacement:", FEM_data)
# print("Latent Representation:", latent_representation)
# print("Autoencoder output:", autoencoder_output)

#Latent space training (NOT USED CURRENTLY) -----------------------------------------------------------------------

if False: #Remove this false statement to include the below code
    l_n = 1
    for k in range(l_n):

        num_epoch = 0

        # Number of training epochs for CAE
        K = 3000
        for i in range(K):
        
            num_epoch += 1

            #print("Reached train_step()")

            loss_latent, u_x_latent, u_latent = train_step_latent()
        
            if num_epoch%2 == 0 :
            
                print("Epoch: ", num_epoch)

            epoch_list.append(num_epoch)

            #lossb = train_step_b()

        #print("pts avg", np.average(pts))
        #print("pts length", len(pts))
        #print("j:", j)
            
        #print("pred a", predictions_a[half - 1])
        #print("pred b", predictions_b[half - 1])

        #print("exact", exact_2[half -1])

#------------------------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))

prediction = []

#for i in predictions:

    #prediction.append(i)

x_plot = np.linspace(0, 1, len(x_data))
x_latent_plot = np.linspace(0, 1, num_latent)

PINN_x_plot = np.linspace(0, 1, num_of_training_pts)

#Autoencoder error very small compared to PDE error, output and input almost overlapping.
ax1.plot(x_latent_plot, latent_representation_vector, label = 'Latent space Representation', color='green')
ax1.plot(x_plot, autoencoder_output_vector, label = 'Autoencoder Representation(Linear autoencoder output)', color='blue', linestyle = "dashed")
ax1.plot(x_plot, x_data, label = '25 node FEM_Solution(Linear autoencoder input)', color='orange')
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

x_true = np.linspace(0, 1, num_of_training_pts)
e = np.e *np.ones(len(points))
exact = -e/(-e**2-1) * e**x_true + e/(-e**2-1)*e**(-x_true)

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
ax1.plot(points, exact, label = "Exact solution", color='purple')

#ax1.plot(epoch_list, loss_list, color = "r")

#ax1.plot(points[:a_pts],u_xa, '--',label = "approx derivative", color='blue')
#ax1.plot(points[b_pts - 1:],u_xb, '--',label = "approx derivative", color='blue')

#ax1.plot(pts,exact_der, label = "exact derivative", color='blue')
#ax1.plot(points[:a_pts],u_xxa, '--',label = "approx second der", color='red')
#ax1.plot(points[b_pts - 1:],u_xxb, '--',label = "approx second der", color='red')

#ax1.plot(pts,exact_sec, label = "exact second der", color='red')
ax1.set(xlabel='x(m)', ylabel='u(m)')
ax1.grid()
ax1.legend()
ax2.plot(epoch_list, loss_list, label = "Loss Histroy", color = "red")
ax2.set(xlabel='Epoch', ylabel='Loss')
ax2.grid()
ax2.legend()
#ax2.plot([(i+1)*10 for i in range(len(losses1))],losses1, label = "Trained Whole", color='blue')

#ax2.set_yscale('log')
#ax2.legend()



plt.show()