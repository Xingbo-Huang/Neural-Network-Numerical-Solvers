# General Math
import numpy as np
import math
from scipy.integrate import odeint
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

num_of_training_pts = 962
N = num_of_training_pts
a_pts = int((N//2))
b_pts = N - a_pts

#points_1 = np.linspace(0, 0.47, 50)
#points_2 = np.linspace(0.47 , 0.53, N - 100)
#points_3 = np.linspace(0.53 , 1, 50)

#points_comb = np.hstack((points_1, points_2, points_3))

#points = tf.convert_to_tensor(points_comb)

points = np.linspace(0, 1, num_of_training_pts)
points_a, points_b = np.split(points, 2)

#points_8  = np.linspace(0.47 + 1/N , 0.53 -1/N , N - 100)

sigmoid_points_a = points_a
sigmoid_points_b = points_b

points = tf.convert_to_tensor(np.hstack((sigmoid_points_a, sigmoid_points_b)))

#points = tf.convert_to_tensor(np.linspace(0, 1, N))

one_a = np.ones(a_pts)
one_b = np.ones(b_pts)

sigmoid_a = -0.5/(one_a + np.exp(5000 * one_a - 10000 * sigmoid_points_a)) + one_a
sigmoid_deriv_a = 5000 * np.power(np.e, -10000 * sigmoid_points_a) /np.power((one_a + np.power(np.e, -10000 * sigmoid_points_a)), 2)

sigmoid_b = -0.5/(one_b + np.exp(5000 * one_b - 10000 * sigmoid_points_b)) + one_b
sigmoid_deriv_b = 5000 * np.power(np.e, -10000 * sigmoid_points_b) /np.power((one_b + np.power(np.e, -10000 * sigmoid_points_b)), 2)

def init_model(num_hidden_layers, num_neurons_per_layer):
    # Initialize a feedforward neural network

    model = tf.keras.Sequential()

    # Input is two-dimensional (time + one spatial dimension)
    model.add(tf.keras.Input(1))

    # Append hidden layers
    for i in range(num_hidden_layers):

        model.add(tf.keras.layers.Dense(num_neurons_per_layer,
        activation=tf.keras.activations.get('tanh'),
        kernel_regularizer= regularizers.L2(),
        bias_regularizer= regularizers.L2()
        ))

    # Output is one-dimensional
    model.add(tf.keras.layers.Dense(1))

    return model

def get_r(model, X_r, boole):
    
    # A tf.GradientTape is used to compute derivatives in TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        # Split t and x to compute partial derivatives
        x = X_r

        # Variables t and x are watched during tape
        # to compute derivatives u_t and u_x
        tape.watch(x)

        # Determine residual 
        u = model(tf.stack(x[:]))

        # Compute gradient u_x within the GradientTape
        # since we need second derivatives
        u_x = tape.gradient(u, x)

        u_xx = tape.gradient(u_x, x)

    u_xxx = tape.gradient(u_xx, x)
    
    del tape

    if boole == True:

        return (sigmoid_a * (u_xx) + x, u_xx, u_x, u, sigmoid_a * (u_xxx + u_x) + sigmoid_deriv_a * (u_xx + x))
    
    if boole == False:

        return (sigmoid_b * (u_xx) + x, u_xx, u_x, u, sigmoid_b * (u_xxx + u_x) + sigmoid_deriv_b * (u_xx + x))

def compute_loss(model):

    r_a,  u_xxa ,u_xa, ua, grad_r_a = get_r(model_a, points[:a_pts], boole = True)
    r_b,  u_xxb ,u_xb, ub, grad_r_a = get_r(model_b, points[b_pts:], boole = False)

    #print("_a, r_b: ------------",r_a, r_b)
    phi_ra = tf.cast(tf.reduce_mean(tf.square(1/(1+abs(u_xa)-u_xa) * r_a)), tf.float32)
    #phi_ra = tf.cast(tf.reduce_mean(tf.square(1/(1+(abs(u_xa)-u_xa))*r_a)), tf.float32)
    bc1 = abs(tf.cast(ua[0], tf.float32))**2
    bc2 = abs(tf.cast(ub[0], tf.float32) - tf.cast(ua[-1], tf.float32))**2
    #grad_enhance = tf.cast(tf.reduce_mean(tf.square(grad_r_a)), tf.float32)
    bc3 = abs(tf.cast(1/2 * u_xb[0], tf.float32) - tf.cast(u_xa[-1], tf.float32))**2

    phi_rb = tf.cast(tf.reduce_mean(tf.square(1/(1+abs(u_xa)-u_xa) * r_b)), tf.float32)
    #phi_rb = tf.cast(tf.reduce_mean(tf.square(r_b)), tf.float32)
    bc4 = abs(tf.cast(u_xb[-1], tf.float32) - 1) ** 2

    loss = phi_ra + phi_rb + bc1 + bc2 + bc4 + bc3 
    #print("loss: ------------",loss)
    
    return loss, u_xa, ua, u_xb, ub, u_xxa, u_xxb

def get_grad(model):
    
    g = 0

    with tf.GradientTape(persistent=True) as tape:
        # This tape is for derivatives with
        # respect to trainable variables
        tape.watch(model.trainable_variables)

        #print("get_grad is working")

        loss, u_xa, ua, u_xb, ub, u_xxa, u_xxb = compute_loss(model)

    g = tape.gradient(loss, model.trainable_variables)
    del tape

    return loss, g, u_xa, ua, u_xb, ub, u_xxa, u_xxb

'''
# Define one training step as a TensorFlow function to increase speed of training
@tf.function
def train_step_a():
    lossa, grad_theta = get_grad(model_a, points_a,boole=True)
    optim_a.apply_gradients(zip(grad_theta, model_a.trainable_variables))

    return lossa

@tf.function
def train_step_b():
    lossb, grad_theta = get_grad(model_b, points_b,boole=False)
    optim_b.apply_gradients(zip(grad_theta, model_b.trainable_variables))

    return lossb
'''
@tf.function
def train_step():

    #print("tran_step() is working")
    
    lossa, grad_theta_a, u_xa, ua, u_xb, ub, u_xxa, u_xxb = get_grad(model_a)
    optim_a.apply_gradients(zip(grad_theta_a, model_a.trainable_variables))
    lossb, grad_theta_b, u_xa, ua, u_xb, ub, u_xxa, u_xxb = get_grad(model_b)
    optim_b.apply_gradients(zip(grad_theta_b, model_b.trainable_variables))

    return lossa, u_xa, ua, u_xb, ub, u_xxa, u_xxb

# Set data type
DTYPE='float32'

tf.keras.backend.set_floatx(DTYPE)

# Initialize model aka u_\theta

optim_a = tf.keras.optimizers.Adam(learning_rate=0.001)
optim_b = tf.keras.optimizers.Adam(learning_rate=0.001)

#tf.keras.backend.clear_session()

model_a = 0
model_a = 0

model_a = init_model(2, 20)
model_b = init_model(2, 20)
model_a.summary()
model_b.summary()
#print("weights:", model_b.layers[1].get_weights()[1])

predictions_a_1 = 0
predictions_b_1 = 0

#tf.random.set_seed(1)

k_n = 1
for k in range(k_n):

    num_epoch = 0

    # Number of training epochs
    M = 1000

    for i in range(M+1):
    
        num_epoch += 1

        #print("Reached train_step()")
        lossa, u_xa, ua, u_xb, ub, u_xxa, u_xxb = train_step()

        if num_epoch%1000 == 0 :
        
            print("Epoch: ", num_epoch)

        #lossb = train_step_b()

    #print("pts avg", np.average(pts))
    #print("pts length", len(pts))
    #print("j:", j)
        
    predictions_a_1 += (1/k_n*model_a(tf.stack([points[:(a_pts)]], axis=1)))
    predictions_b_1 += (1/k_n*model_b(tf.stack([points[(b_pts):]], axis=1)))
        
    #print("pred a", predictions_a[half - 1])
    #print("pred b", predictions_b[half - 1])

    #print("exact", exact_2[half -1])

#------------------------------------------------------------------------------------------

fig, (ax1) = plt.subplots(1,1,figsize=(6,6))

prediction = []

for i in predictions_a_1:

    prediction.append(i)

for i in predictions_b_1:

    prediction.append(i)

points = np.hstack((sigmoid_points_a, sigmoid_points_b))
points_2 = np.linspace(0,1,N)

ax1.plot(points, prediction, label = 'WE-GE-PINN solution with uniform input', color='green')
#ax1.plot(points_2, prediction_2, label = 'WE-GE-PINN solution with uniform distribution input', color='red')
#ax1.plot(points[b_pts:],predictions_b, label = 'Approx Full', color='green')

#exact_1 = (points[:a_pts]**4/24)
#exact_2 = (points[b_pts:]**2/8 - 1/24)
#exact   = np.append(exact_1,exact_2)

#exact = -points*points*points/6 + 3*points/2

exact_1 = -points[:a_pts]*points[:a_pts]*points[:a_pts]/6 + 8*points[:a_pts]/8
exact_2 = -points[a_pts:]*points[a_pts:]*points[a_pts:]/3 + 2*points[a_pts:] - 23/48
exact   = np.append(exact_1,exact_2)

residual = 0

for point in range(1, a_pts - 1, 1):

    #L1 residual
    residual += float(1/len(points))* (np.square((exact_2[point]-predictions_b_1[point])) + np.square((exact_1[point]-predictions_a_1[point])))

    #L2 residual
    #esidual += float(1/len(points))* (np.square((exact_2[point]-predictions_b_1[point])/exact_2[point]) + np.square((exact_1[point]-predictions_a_1[point])/exact_1[point]))

print(residual)

#exact_1_der = -pts[:half+1]**2/2 + 2
#exact_2_der = -pts[half+1:]**2 + 2
#exact_der   = np.append(exact_1_der,exact_2_der)

#exact_1_sec = -pts[:half+1]
#exact_2_sec = -(1/A)*pts[half+1:]
#exact_sec   = np.append(exact_1_sec,exact_2_sec)
ax1.plot(points, exact, label = "Exact solution", color='blue')

#ax1.plot(points[:a_pts],u_xa, '--',label = "approx derivative", color='blue')
#ax1.plot(points[b_pts - 1:],u_xb, '--',label = "approx derivative", color='blue')

#ax1.plot(pts,exact_der, label = "exact derivative", color='blue')
#ax1.plot(points[:a_pts],u_xxa, '--',label = "approx second der", color='red')
#ax1.plot(points[b_pts - 1:],u_xxb, '--',label = "approx second der", color='red')

#ax1.plot(pts,exact_sec, label = "exact second der", color='red')
ax1.set(xlabel='x (m)', ylabel='u')

ax1.legend()
#ax2.plot([(i+1)*10 for i in range(len(losses1))],losses1, label = "Trained Whole", color='blue')

#ax2.set_yscale('log')
#ax2.legend()

plt.show()