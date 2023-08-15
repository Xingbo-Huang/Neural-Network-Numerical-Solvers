# General Math
import numpy as np
import math
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

num_of_training_pts = 200
N = num_of_training_pts
a_pts = int((N//2))
b_pts = N - a_pts

discontinuity = 0.5
sparce = 20

x_nodes = 20
delta_h = 1/(x_nodes - 1)
nodes_per_subdomain = num_of_training_pts//x_nodes

x_node_list = np.linspace(1, x_nodes - 2, x_nodes - 2)
gloabl_x_distribution = np.linspace(0, 1, x_nodes + 1)

run_count = 0

#Trial function = test function defintion-------------------------------------------------------------------------

def local_trial_func(global_x_distribution, N, node_num):

    if node_num == 0:

        xa = -1/x_nodes
        xc = global_x_distribution[node_num]
        xb = global_x_distribution[node_num + 1]

        local_x_distribution_mid_1 = (np.linspace(xa, xc, N//2))
        local_x_distribution_mid_2 = (np.linspace(xc + (xb-xc)/(N - N//2 - 1), xb, N - N//2 - 1))
        local_x_distribution = np.hstack((local_x_distribution_mid_1, local_x_distribution_mid_2))

    elif node_num == len(global_x_distribution) -1 :

        xa = global_x_distribution[node_num - 1]
        xc = global_x_distribution[node_num]
        xb = xc + 1/x_nodes

        local_x_distribution_mid_1 = (np.linspace(xa, xc, N//2))
        local_x_distribution_mid_2 = (np.linspace(xc + (xb-xc)/(N - N//2 - 1), xb, N - N//2 - 1))
        local_x_distribution = np.hstack((local_x_distribution_mid_1, local_x_distribution_mid_2))

    else:

        xa = global_x_distribution[node_num - 1]
        xc = global_x_distribution[node_num]
        xb = global_x_distribution[node_num + 1]

        local_x_distribution_mid_1 = (np.linspace(xa, xc, N//2))
        local_x_distribution_mid_2 = (np.linspace(xc + (xb-xc)/(N - N//2 - 1), xb, N - N//2 - 1))
        local_x_distribution = np.hstack((local_x_distribution_mid_1, local_x_distribution_mid_2))

    x_bar = []
    trial_func_deriv = []

    if True:

        for i in local_x_distribution:

            if xa <= i <= xc:

                x_bar_i = (i-xc)/((xc-xa))
                x_bar.append(x_bar_i + 1)
            
            elif xc < i <= xb:

                x_bar_i = (i-xc)/((xb-xc))
                x_bar.append(1 - x_bar_i)

        trial_func = x_bar 

        for i in local_x_distribution:

            if xa <= i <= xc:

                x_bar_i_deriv = 1/ delta_h
                trial_func_deriv.append(x_bar_i_deriv)
            
            elif xc < i <= xb:

                x_bar_i_deriv = -1/delta_h
                trial_func_deriv.append(x_bar_i_deriv)

    trial_func_2nd_deriv = 0

    return trial_func, trial_func_deriv, trial_func_2nd_deriv, local_x_distribution

#---------------------------------------------------------------------------------------------------
def local_test_func(global_x_distribution, N, node_num):

    if node_num == 0:

        xa = -1
        xc = global_x_distribution[node_num]
        xb = global_x_distribution[node_num + 1]

        local_x_distribution_mid_1 = (np.linspace(xa, xc, N//2))
        local_x_distribution_mid_2 = (np.linspace(xc + (xb-xc)/(N - N//2 - 1), xb, N - N//2 - 1))
        local_x_distribution = np.hstack((local_x_distribution_mid_1, local_x_distribution_mid_2))

    elif node_num == len(global_x_distribution) -1 :

        xa = global_x_distribution[node_num - 1]
        xc = global_x_distribution[node_num]
        xb = len(global_x_distribution) 

        local_x_distribution_mid_1 = (np.linspace(xa, xc, N//2))
        local_x_distribution_mid_2 = (np.linspace(xc + (xb-xc)/(N - N//2 - 1), xb, N - N//2 - 1))
        local_x_distribution = np.hstack((local_x_distribution_mid_1, local_x_distribution_mid_2))
        
    else:

        xa = global_x_distribution[node_num - 1]
        xc = global_x_distribution[node_num]
        xb = global_x_distribution[node_num + 1]

        local_x_distribution_mid_1 = (np.linspace(xa, xc, N//2))
        local_x_distribution_mid_2 = (np.linspace(xc + (xb-xc)/(N - N//2 - 1), xb, N - N//2 - 1))
        local_x_distribution = np.hstack((local_x_distribution_mid_1, local_x_distribution_mid_2))

    x_bar = []
    test_func_deriv = []
    test_func_sec_deriv = []

    if node_num != 0 and node_num != len(global_x_distribution) - 1:

        for i in local_x_distribution:

            if xa <= i <= xc:

                x_bar_i = (i-xc)/((xc-xa))
                x_bar.append(x_bar_i)
            
            elif xc < i <= xb:

                x_bar_i = (i-xc)/((xb-xc))
                x_bar.append(x_bar_i)

        test_func = np.square((np.square(x_bar) - 1))

        for i in local_x_distribution:

            if xa <= i <= xc:

                x_bar_i_deriv = 2 * (np.square((i-xc)/((xc-xa)))-1) * 2 * (i-xc)/np.square((xc-xa))
                test_func_deriv.append(x_bar_i_deriv)
            
            elif xc < i <= xb:

                x_bar_i_deriv = 2 * (np.square((i-xc)/((xb-xc)))-1)* 2 * (i-xc)/np.square((xb-xc))
                test_func_deriv.append(x_bar_i_deriv)

        #Change################################################
        for i in local_x_distribution:

            if xa <= i <= xc:

                x_bar_i_sec_deriv = 1/((xc-xa)**4) * 4 * (3 * i ** 2 - 6 * xc * i + 2 * xc ** 2 - xa ** 2 + 2 * xc * xa)
                test_func_sec_deriv.append(x_bar_i_sec_deriv)
            
            elif xc < i <= xb:

                x_bar_i_sec_deriv = 1/((xb-xc)**4) * 4 * (3 * i ** 2 - 6 * xc * i + 2 * xc ** 2 - xb ** 2 + 2 * xc * xb)
                test_func_sec_deriv.append(x_bar_i_sec_deriv)

    elif node_num == 0:

        for i in local_x_distribution:
        
            x_bar_i = (i-xc)/((xb-xc))
            x_bar.append(x_bar_i)

        test_func = np.square((np.square(x_bar) - 1))

        for i in local_x_distribution:

            x_bar_i_deriv = 2 * np.square(np.square((i-xc)/((xc-xb)))-1) * 2 * (i-xc)/np.square((xc-xb))
            test_func_deriv.append(x_bar_i_deriv)

        for i in local_x_distribution:

            x_bar_i_sec_deriv = 1/((xb-xc)**4) * 4 * (3 * i ** 2 - 6 * xc * i + 2 * xc ** 2 - xb ** 2 + 2 * xc * xb)
            test_func_sec_deriv.append(x_bar_i_sec_deriv)

    elif node_num == len(global_x_distribution)-1:

        for i in local_x_distribution:
        
            x_bar_i = (i-xc)/((xc-xa))
            x_bar.append(x_bar_i)

        test_func = np.square((np.square(x_bar) - 1))

        for i in local_x_distribution:

            x_bar_i_deriv = 2 * np.square(np.square((i-xc)/((xc-xa)))-1) * 2 * (i-xc)/np.square((xc-xa))
            test_func_deriv.append(x_bar_i_deriv)

        for i in local_x_distribution:

            x_bar_i_sec_deriv = 1/((xc-xa)**4) * 4 * (3 * i ** 2 - 6 * xc * i + 2 * xc ** 2 - xa ** 2 + 2 * xc * xa)
            test_func_sec_deriv.append(x_bar_i_sec_deriv)

    return test_func, test_func_deriv, test_func_sec_deriv, local_x_distribution
#-----------------------------------------------------------------------------------------------------

#Sigmoid evaluation for both neural networks--------------------------------------------------------------

one_a = np.ones(a_pts)
one_b = np.ones(b_pts)

#points_4 = np.linspace(0, discontinuity - 0.03, sparce)
#points_5 = np.linspace(discontinuity - 0.03 + 1/(100*(a_pts - sparce)) , discontinuity, a_pts - sparce)
#points_6 = np.linspace(discontinuity + 1/(100*(b_pts - sparce)), discontinuity + 0.03, b_pts - sparce)
#points_7 = np.linspace(discontinuity + 0.03 + 1/50 , 1, sparce)

#sigmoid_points_a = np.hstack((points_4, points_5))
#sigmoid_points_b = np.hstack((points_6, points_7))

#points = tf.cast(tf.convert_to_tensor(np.hstack((sigmoid_points_a, sigmoid_points_b))), tf.float32)

points = np.linspace(0, 1, num_of_training_pts)
points_a, points_b = np.split(points, 2)

sigmoid_points_a = points_a
sigmoid_points_b = points_b

points = tf.convert_to_tensor(np.hstack((sigmoid_points_a, sigmoid_points_b)))

sigmoid_a = 0.5/(one_a + np.power(np.e, one_a * 5000 - 10000 * sigmoid_points_a))
sigmoid_deriv_a = 5000 * np.power(np.e, -10000 * sigmoid_points_a) /np.power((one_a + np.power(np.e, -1000 * sigmoid_points_a)), 2)

sigmoid_b = 0.5/(one_b + np.power(np.e, one_b * 5000 -10000 * sigmoid_points_b))
sigmoid_deriv_b = 5000 * np.power(np.e, -10000 * sigmoid_points_b) /np.power((one_b + np.power(np.e, -1000 * sigmoid_points_b)), 2)

#Simpsons integration method, midpoint is interpolated-------------------------------------------------

def simpson_integrate(x_list, y_list, N):

    area = 0

    for i in range(0, N-1):

        dx = x_list[i+1] - x_list[i]
        dx_mid = dx/2

        dA = (y_list[i] + 4 * (y_list[i] + 0.5 * (y_list[i+1] - y_list[i])/(dx)) + y_list[i+1]) * dx/6
        area += dA

    return area

#Neural network functions--------------------------------------------------------------------------------

def init_model(num_hidden_layers, num_neurons_per_layer):
    # Initialize a feedforward neural network

    model = tf.keras.Sequential()

    # Input is two-dimensional (time + one spatial dimension)
    model.add(tf.keras.Input(1))

    # Append hidden layers
    for i in range(num_hidden_layers):

        model.add(tf.keras.layers.Dense(num_neurons_per_layer,
        activation=tf.keras.activations.get('sigmoid'),
        kernel_regularizer= regularizers.L2(),
        bias_regularizer= regularizers.L2()
        ))

    # Output is one-dimensional
    model.add(tf.keras.layers.Dense(1, name = "output"))

    return model

def get_r(model, X_r):

    x = tf.stack(X_r[:])
  
    # A tf.GradientTape is used to compute derivatives in TensorFlow
    with tf.GradientTape(persistent = True) as tape:
        # Split t and x to compute partial derivatives
        
        # Variables t and x are watched during tape
        # to compute derivatives u_t and u_x
        tape.watch(x)

        # Determine residual 
        u = model(x)

        # Compute gradient u_x within the GradientTape
        # since we need second derivatives
        u_x = tape.gradient(u, x)

    u_xx = tape.gradient(u_x, x)
    # print("u_x:", u_x, "u_xx:", u_xx)

    print(u)
    print(u_x)
    print(u_xx)

    u = tf.cast(tf.squeeze(u), tf.float64)
    
    del tape

    return ((u_xx - u) - u_x, u_xx, u_x, u, x)
    
def compute_loss(model):

    PDE_loss_weak_1 = 0
    PDE_loss_weak_2 = 0

    PDE_loss_weak_u_x_phi_x = 0
    PDE_loss_weak_u_xx_phi= 0
    PDE_loss_weak_x_phi = 0

    #uniform = tf.cast(tf.convert_to_tensor(np.linspace(0, 1, 2001)), tf.float32)
    
    r_a,  u_xxa ,u_xa, ua, x_a = get_r(model, points)

    ua = tf.squeeze(ua)
    ua = tf.cast(ua, tf.float32)

    print("Runnning --------------------------------------------------")

    for i in x_node_list:
        
        u_xxa_i = u_xxa[int((nodes_per_subdomain * i) - (nodes_per_subdomain)) : int((nodes_per_subdomain * i + (nodes_per_subdomain)))]
        u_xa_i = u_xa[int((nodes_per_subdomain * i) - (nodes_per_subdomain)) : int((nodes_per_subdomain * i + (nodes_per_subdomain)))]
        u_a_i = tf.cast(ua[int((nodes_per_subdomain * i) - (nodes_per_subdomain)) : int((nodes_per_subdomain * i + (nodes_per_subdomain)))], tf.float64)
        r_a_i = r_a[int((nodes_per_subdomain * i) - (nodes_per_subdomain)) : int((nodes_per_subdomain * i + (nodes_per_subdomain)))]    
        x_a_i = x_a[int((nodes_per_subdomain * i) - (nodes_per_subdomain)) : int((nodes_per_subdomain * i + (nodes_per_subdomain)))]

        trial_func, trial_func_deriv, trial_func_2nd_deriv, local_x_distribution = local_test_func(gloabl_x_distribution, 2 * nodes_per_subdomain+1, int(i))
        
        # print("u_x", u_xa_i)
        # print("phi_x", trial_func_deriv)

        # print("----------------------------------------------------------")

        # print("u_xx",u_xxa_i)
        # print("phi", trial_func)
        #integrand = -u_xa_i * trial_func_deriv + x_a_i * trial_func

        if i == x_node_list[-1]:
            
            integrand1 = -u_xa_i * trial_func_deriv + x_a_i * trial_func 
            # integrand1 = u_a_i * trial_func_2nd_deriv + x_a_i * trial_func 
            # integrand2 = (1 * u_xxa_i + x_a_i) * trial_func 
            # integrand2 = (1 * u_xxa_i - u_xa_i) * trial_func 
            # integrand2 = (u_xa_i - x_a_i) * trial_func 
            # integrand2 = (1 * u_xxa_i - u_a_i) * trial_func 
            # integrand2 = (1 * u_xa_i - u_a_i) * trial_func 
            # integrand2 = (r_a_i) * trial_func 

        else:
            
            integrand1 = -u_xa_i * trial_func_deriv + x_a_i * trial_func 
            # integrand1 = u_a_i * trial_func_2nd_deriv + x_a_i * trial_func 
            # integrand2 = (1 * u_xxa_i + x_a_i) * trial_func
            # integrand2 = (1 * u_xxa_i - u_xa_i) * trial_func 
            # integrand2 = (u_xa_i - x_a_i) * trial_func 
            # integrand2 = (1 * u_xxa_i - u_a_i) * trial_func 
            # integrand2 = (1 * u_xa_i - u_a_i) * trial_func 
            # integrand2 = (r_a_i) * trial_func 

        integrand_tensor1 = tf.cast(tf.convert_to_tensor(integrand1), tf.float32)
        # integrand_tensor2 = tf.cast(tf.convert_to_tensor(integrand2), tf.float32)

        # integrand_u_x_phi_x = tf.cast(tf.convert_to_tensor(-u_xa_i * trial_func_deriv), tf.float32)
        # integrand_u_xx_phi = tf.cast(tf.convert_to_tensor((1 * u_xxa_i) * trial_func), tf.float32)
        # integrand_x_phi = tf.cast(tf.convert_to_tensor((x_a_i * trial_func) * trial_func), tf.float32)

        local_x_distribution = tf.cast(tf.convert_to_tensor(local_x_distribution), tf.float32)

        #loss_i = simpson_integrate(local_x_distribution, integrand_tensor, nodes_per_subdomain)
        loss_i_1 = simpson_integrate(local_x_distribution, integrand_tensor1, nodes_per_subdomain)
        # loss_i_2 = simpson_integrate(local_x_distribution, integrand_tensor2, nodes_per_subdomain)

        # integral_u_x_phi_x = simpson_integrate(local_x_distribution, integrand_u_x_phi_x, nodes_per_subdomain)
        # integral_u_xx_phi = simpson_integrate(local_x_distribution, integrand_u_xx_phi, nodes_per_subdomain)
        # integral_x_phi = simpson_integrate(local_x_distribution, integrand_x_phi, nodes_per_subdomain)

        # print("U_x Phi_x component (1):", integral_u_x_phi_x)
        # print("x Phi component (1): ", integral_x_phi)

        # print("\n")

        # print("U_xx Phi component (2) :", integral_u_xx_phi)

        # print("\n")

        # print("Integral result 1", loss_i_1)
        # print("Integral result 2", loss_i_2)
        # print("---------------------------------------------------")

        PDE_loss_weak_1 += 1/(delta_h) * loss_i_1 ** 2
        # PDE_loss_weak_2 += 1/(delta_h) * loss_i_2 ** 2

        # PDE_loss_weak_u_x_phi_x += 1/(delta_h) * integral_u_x_phi_x ** 2
        # PDE_loss_weak_u_xx_phi += 1/(delta_h) * integral_u_xx_phi ** 2
        # PDE_loss_weak_x_phi += 1/(delta_h) * integral_x_phi ** 2
        
    # print("PDE Loss 1:", PDE_loss_weak_1)
    # print("PDE Loss 2:", PDE_loss_weak_2)

    # print("PDE Loss u_x phi_x:", PDE_loss_weak_u_x_phi_x)
    # print("PDE Loss u_xx phi:", PDE_loss_weak_u_xx_phi)
    # print("PDE Loss x phi:", PDE_loss_weak_x_phi)
    
    phi_ra = tf.cast(tf.reduce_mean(tf.square(r_a)), tf.float32)
    #phi_ra = tf.cast(tf.reduce_mean(tf.square(1/(1+abs(u_xa)-u_xa)*r_a)), tf.float32)

    # bc1 = abs(tf.cast(ua[0], tf.float32))**2
    bc1 = abs(tf.cast(ua[0], tf.float32) - 0.)**2

    #bc2 = abs(tf.cast(ub[0], tf.float32) - tf.cast(ua[-1], tf.float32))**2
    #bc3 = tf.cast(tf.reduce_mean(tf.square(grad_r_a)), tf.float32)
    #bc3 = abs(tf.cast(1 * u_xb[0], tf.float32) - tf.cast(u_xa[-1], tf.float32))**2

    #phi_rb = tf.cast(tf.reduce_mean(tf.square(r_b)), tf.float32)
    #phi_rb = tf.cast(tf.reduce_mean(tf.square(1/(1+abs(u_xb)-u_xb)*r_b)), tf.float32)
    bc4 = abs(tf.cast(u_xa[-1], tf.float32) - 1.)**2

    true_u  = tf.cast(-points*points*points/6 + 3*points/2, tf.float32)

    data = tf.reduce_mean((ua[:300] - true_u[:300])**2)

    loss = (PDE_loss_weak_1 * delta_h + bc1 + bc4)
    # loss = (PDE_loss_weak_2 + bc1)
    # loss = (100 * phi_ra + bc1)
    #print("loss: ------------",loss)
    
    return loss, u_xa, ua

def get_grad(model):
    
    g = 0

    with tf.GradientTape(persistent=True) as tape:
        # This tape is for derivatives with
        # respect to trainable variables
        tape.watch(model.trainable_variables)

        #print("get_grad is working")

        loss, u_xa, ua = compute_loss(model)

    g = tape.gradient(loss, model.trainable_variables)
    del tape

    return loss, g, u_xa, ua

@tf.function
def train_step():

    #print("tran_step() is working")
    
    lossa, grad_theta_a, u_xa, ua = get_grad(model_a)
    optim_a.apply_gradients(zip(grad_theta_a, model_a.trainable_variables))

    return lossa, u_xa, ua

#Running the neural network----------------------------------------------------------------------------

# Set data type
DTYPE='float32'

tf.keras.backend.set_floatx(DTYPE)

# Initialize model aka u_\theta

optim_a = tf.keras.optimizers.Adam(learning_rate=0.0001)

#tf.keras.backend.clear_session()

model_a = 0

model_a = init_model(2, 50)

#model_b.summary()
#print("weights:", model_b.layers[1].get_weights()[1])

predictions = 0

#tf.random.set_seed(1)
a, b, d, c = local_test_func(gloabl_x_distribution, 2 * nodes_per_subdomain, 1)
plt.grid()
plt.plot(c, b)
plt.plot(c, d)
plt.plot(c, a)
plt.show()

loss_list = []
epoch_list = []

k_n = 1
for k in range(k_n):

    num_epoch = 0

    # Number of training epochs
    M = 10000
    for i in range(M):
    
        num_epoch += 1

        #print("Reached train_step()")

        lossa, u_xa, ua = train_step()
    
        if num_epoch%2 == 0 :
        
            print("Epoch: ", num_epoch)

        loss_list.append(lossa)
        epoch_list.append(num_epoch)

        #lossb = train_step_b()

    #print("pts avg", np.average(pts))
    #print("pts length", len(pts))
    #print("j:", j)
        
    predictions += (1/k_n*model_a(tf.stack([points], axis=1)))
        
    #print("pred a", predictions_a[half - 1])
    #print("pred b", predictions_b[half - 1])

    #print("exact", exact_2[half -1])

#------------------------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))

prediction = []

for i in predictions:

    prediction.append(i)

# Writing solution to file-------------------------------
# Assume you have the output tensor named 'output_tensor'
output_tensor = predictions

# Save the output tensor
model_a.save(r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\NN_Models\1NN_Strong_PINN_model")

#-----------------------------------------------------------

points = np.hstack((sigmoid_points_a, sigmoid_points_b))
points_2 = np.linspace(0,1,N)

ax1.plot(points, prediction, label = 'Weak-PINN solution with uniform input', color='green')
#ax1.plot(points[b_pts:],predictions_b, label = 'Approx Full', color='green')

#exact_1 = (points[:a_pts]**4/24)
#exact_2 = (points[b_pts:]**2/8 - 1/24)
#exact   = np.append(exact_1,exact_2)

#exact = -points*points*points/6 + 3*points/2

exact_1 = -points[:a_pts]*points[:a_pts]*points[:a_pts]/6 + 3*points[:a_pts]/2
exact_2 = -points[b_pts:]*points[b_pts:]*points[b_pts:]/6 + 3*points[b_pts:]/2
exact = np.hstack((exact_1, exact_2))

#exact_1 = -points[:a_pts]*points[:a_pts]*points[:a_pts]/6 + 8*points[:a_pts]/8
#exact_2 = -points[b_pts:]*points[b_pts:]*points[b_pts:]/3 + 2*points[b_pts:] - 23/48
exact   = np.append(exact_1,exact_2)

residual = 0

for point in range(1, num_of_training_pts - 2, 1):

    #L1 residual
    #residual += float(1/len(points))* (np.square((exact_2[point]-predictions_b_1[point])) + np.square((exact_1[point]-predictions_a_1[point])))

    #L2_residual
    residual += float(1/len(points))* np.square((exact[point]-predictions[point])/exact[point])

print("Residual:", residual)

#exact_1_der = -pts[:half+1]**2/2 + 2
#exact_2_der = -pts[half+1:]**2 + 2
#exact_der   = np.append(exact_1_der,exact_2_der)

#exact_1_sec = -pts[:half+1]
#exact_2_sec = -(1/A)*pts[half+1:]
#exact_sec   = np.append(exact_1_sec,exact_2_sec)
ax1.plot(points, exact, label = "Exact solution", color='blue')
ax2.plot(epoch_list, loss_list, color = "r", label = "Loss evolution")
ax2.set_yscale('log')

#ax1.plot(points[:a_pts],u_xa, '--',label = "approx derivative", color='blue')
#ax1.plot(points[b_pts - 1:],u_xb, '--',label = "approx derivative", color='blue')

#ax1.plot(pts,exact_der, label = "exact derivative", color='blue')
#ax1.plot(points[:a_pts],u_xxa, '--',label = "approx second der", color='red')
#ax1.plot(points[b_pts - 1:],u_xxb, '--',label = "approx second der", color='red')

#ax1.plot(pts,exact_sec, label = "exact second der", color='red')
ax1.set(xlabel='x (m)', ylabel='u')
ax1.grid()
ax1.legend()
ax2.set(xlabel='Epoch', ylabel='Loss')
ax2.grid()
ax2.legend()
#ax2.plot([(i+1)*10 for i in range(len(losses1))],losses1, label = "Trained Whole", color='blue')

#ax2.set_yscale('log')
#ax2.legend()

plt.show()