# FEM Traditional method implementation for 1D Bar
# Started 2023-05-30
# Author: Xingbo Huang

# PDE: u_xx + x = 0

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#Problem Initialization-----------------------

x_nodes = 25
delta_h = 1/(x_nodes-1)
x_length = 1

A_matrix = np.empty((x_nodes, x_nodes))
b_matrix = np.empty(x_nodes)

x_distribution = np.linspace(0, 1, x_nodes)

#Trial function evaluation--------------------------------------------------------

#Determine x_bar and the associated weight function and its derivative i.e. (w and w_x)

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

                x_bar_i_deriv = 1
                trial_func_deriv.append(x_bar_i_deriv)
            
            elif xc < i <= xb:

                x_bar_i_deriv = -1
                trial_func_deriv.append(x_bar_i_deriv)

    return trial_func, trial_func_deriv, local_x_distribution

#---------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

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

                x_bar_i_deriv = 2 * np.square(np.square((i-xc)/((xc-xa)))-1) * 2 * (i-xc)/np.square((xc-xa))
                test_func_deriv.append(x_bar_i_deriv)
            
            elif xc < i <= xb:

                x_bar_i_deriv = 2 * np.square(np.square((i-xc)/((xc-xa)))-1) * 2 * (i-xc)/np.square((xc-xa))
                test_func_deriv.append(x_bar_i_deriv)

        #Change################################################
        for i in local_x_distribution:

            if xa <= i <= xc:

                x_bar_i_sec_deriv = 1/((xc-xa)**4) * 4 * (3 * i ** 2 - 6 * xc * i + 2 * xc ** 2 - xa ** 2 + 2 * xc * xa)
                test_func_sec_deriv.append(x_bar_i_sec_deriv)
            
            elif xc < i <= xb:

                x_bar_i_sec_deriv = x_bar_i_sec_deriv = 1/((xb-xc)**4) * 4 * (3 * i ** 2 - 6 * xc * i + 2 * xc ** 2 - xb ** 2 + 2 * xc * xb)
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

def Apply_BC(A, b):

    #Left BC
    #u(0) = 0

    for i in range(x_nodes):
        if i == 0:

            A[0, i] = 1.0000000
            b[i] = 0.00000000

        else:

            A[0, i] = 0

    #Right BC
    #u_x(1) = 1 , equivalent to
    #u_{n-1} - u_{n} = 1

    for i in range(x_nodes):

        if i == x_nodes - 1:

            A[x_nodes - 1, i] = 1.00000000

            b[x_nodes - 1] = 1 * delta_h

        elif i == x_nodes - 2:

            A[x_nodes - 1, i] = -1.00000000

        else:  

            A[x_nodes - 1, i] = 0

    return A, b

def Apply_BC2(A, b):

    #Left BC
    #u(0) = 0

    for i in range(x_nodes):
        if i == 0:

            A[0, i] = 1.0000000
            b[i] = 0.00000000

        else:

            A[0, i] = 0

    #Right BC
    #u_x(1) = 1 , equivalent to
    #u_{n-1} - u_{n} = 1

    #FEM BC
    b[-1] = (b[-1] - 1) 

    #FDM BC
    # A[x_nodes -1, x_nodes - 2] = -1
    # A[x_nodes -1, x_nodes - 1] = 1
    # b[x_nodes - 1] = 1 * delta_h

    return A, b

#Interval definition
global_x = np.linspace(0, 1, x_nodes)

#test_func, test_func_deriv, test_func_sec_deriv, local_x_distribution = local_test_func(global_x, 10, 3)

trial_func, trial_func_deriv, local_x_distribution = local_trial_func(global_x, 30, 2)

#Test functions
#plt.plot(local_x_distribution, test_func, "g")
#plt.plot(local_x_distribution, test_func_deriv, "r")
#plt.plot(local_x_distribution, test_func_sec_deriv, "r")

#Trial functions
plt.grid()
plt.plot(local_x_distribution, trial_func, "ro", "g")
plt.plot(local_x_distribution, trial_func_deriv, "r")

plt.show()                                                                                                                                                                                                          

#Simpsons integration. Nodes must be multiple of 2
def simpson_integrate(x_list, y_list, N):

    area = 0

    for i in range(0, N-2, 2):

        dx = x_list[i+2] - x_list[i]

        dA =  (y_list[i] + 4 * y_list[i+1] + y_list[i+2]) * dx/6
        area += dA

    return area

#Generate matrix-------------------------------------------------------------------

node_list = np.linspace(0, x_nodes - 1, x_nodes)

element_1_nodes = []
element_2_nodes = []

nodes_per_subdomain = 100000
nodes_per_element = nodes_per_subdomain//2

print(node_list)

for i in node_list:

    print("i", i)

    if i == 0: 

        N_1_i, N_1_i_x, r,  local_x_distribution = local_test_func(global_x, nodes_per_subdomain, 0)
        N_2_i, N_2_i_x, g, local_x_distribution = local_test_func(global_x, nodes_per_subdomain, 1)

        N_1_i, N_1_i_x, r, local_x_distribution = np.array(N_1_i), np.array(N_1_i_x), np.array(r), np.array(local_x_distribution)
        N_2_i, N_2_i_x, g, local_x_distribution = np.array(N_2_i), np.array(N_2_i_x), np.array(g), np.array(local_x_distribution)

        test_func, test_func_deriv, test_func_sec_deriv, local_x_distribution = local_test_func(global_x, nodes_per_subdomain, 0)
        test_func, test_func_deriv, test_func_sec_deriv, local_x_distribution = np.array(test_func), np.array(test_func_deriv), np.array(test_func_sec_deriv), np.array(local_x_distribution)

        element_1_x_vals = local_x_distribution[:nodes_per_element]
        element_2_x_vals = local_x_distribution[nodes_per_element:]

        #checking purposes
        element_1_num_pts = len(element_1_x_vals)
        element_2_num_pts = len(element_2_x_vals)

        #N_2_i has both element points but for node 0 we only use element 1 points
        integrand_c0 = N_1_i[0 : element_1_num_pts] * test_func_sec_deriv[0 : element_1_num_pts]
        integrand_c1 = N_2_i[0 : element_1_num_pts] * test_func_sec_deriv[0 : element_1_num_pts]

        inetgrand_b0 = element_1_x_vals * test_func[:element_1_num_pts]

        c0 = simpson_integrate(element_1_x_vals, integrand_c0, element_1_num_pts)
        c1 = simpson_integrate(element_1_x_vals, integrand_c1, element_1_num_pts)
        b0 = simpson_integrate(element_1_x_vals, inetgrand_b0, element_1_num_pts)

        for i in range(x_nodes):

            if i == 0:
                
                A_matrix[0, i] = c0

            elif i == 1:
                
                A_matrix[0, i] = c1

            else:

                A_matrix[0, i] = 0

        b_matrix[0] = b0

    #Runs well till here ###################################################################

    #This section is kinda fixed??????????
    #Edit this (currently copied from first case)
    elif i == (x_nodes - 1):

        i = int(i)

        print("running")

        N_1_i_min1, N_1_i_min1_x, local_x_distribution = local_trial_func(global_x, nodes_per_subdomain, i-1)
        N_2_i_min1, N_2_i_min1_x, local_x_distribution = local_trial_func(global_x, nodes_per_subdomain, i)

        N_1_i_min1, N_1_i_min1_x, local_x_distribution = np.array(N_1_i_min1), np.array(N_1_i_min1_x), np.array(local_x_distribution)
        N_2_i_min1, N_2_i_min1_x, local_x_distribution = np.array(N_2_i_min1), np.array(N_2_i_min1_x), np.array(local_x_distribution)

        test_func, test_func_deriv, local_x_distribution = local_trial_func(global_x, nodes_per_subdomain, i)
        test_func, test_func_deriv, local_x_distribution = np.array(test_func), np.array(test_func_deriv), np.array(local_x_distribution)

        element_1_x_vals = local_x_distribution[:nodes_per_element]
        element_2_x_vals = local_x_distribution[nodes_per_element:]

        #checking purposes
        element_1_num_pts = len(element_1_x_vals)
        element_2_num_pts = len(element_2_x_vals)

        integrand_c0 = N_1_i_min1_x[element_2_num_pts:] * 1 
        integrand_c1 = N_2_i_min1_x[:element_2_num_pts] * 1 

        inetgrand_b0 = element_1_x_vals * test_func[:element_1_num_pts:]

        c0 = simpson_integrate(element_1_x_vals, integrand_c0, element_1_num_pts) /delta_h ** 2
        c1 = simpson_integrate(element_1_x_vals, integrand_c1, element_1_num_pts) /delta_h ** 2
        b0 = simpson_integrate(element_1_x_vals, inetgrand_b0, element_1_num_pts)

        for i in range(x_nodes):

            if i == x_nodes -2:
                
                A_matrix[i + 1, i] = -c0

            elif i == x_nodes - 1:
                
                A_matrix[i, i] = -c1

            else:
                print("running")
                A_matrix[x_nodes - 1, i] = 0

        b_matrix[i] = -b0 
    #Check this section #####################################################################
    else: 

        i = int(i)

        N_1_i_min1, N_1_i_min1_x, local_x_distribution = local_trial_func(global_x, nodes_per_subdomain, i-1)
        N_2_i_min1, N_2_i_min1_x, local_x_distribution = local_trial_func(global_x, nodes_per_subdomain, i)
        N_1_i, N_1_i_x, local_x_distribution = local_trial_func(global_x, nodes_per_subdomain, i)
        N_2_i, N_2_i_x, local_x_distribution = local_trial_func(global_x, nodes_per_subdomain, i+1)

        N_1_i_min1, N_1_i_min1_x, local_x_distribution = np.array(N_1_i_min1), np.array(N_1_i_min1_x), np.array(local_x_distribution)
        N_2_i_min1, N_2_i_min1_x, local_x_distribution = np.array(N_2_i_min1), np.array(N_2_i_min1_x), np.array(local_x_distribution)
        N_1_i, N_1_i_x, local_x_distribution = np.array(N_1_i), np.array(N_1_i_x), np.array(local_x_distribution)
        N_2_i, N_2_i_x, local_x_distribution = np.array(N_2_i), np.array(N_2_i_x), np.array(local_x_distribution)

        test_func, test_func_deriv, local_x_distribution = local_trial_func(global_x, nodes_per_subdomain, i)
        test_func, test_func_deriv, local_x_distribution = np.array(test_func), np.array(test_func_deriv), np.array(local_x_distribution)

        element_1_x_vals = local_x_distribution[:nodes_per_element]
        element_2_x_vals = local_x_distribution[nodes_per_element:]

        #checking purposes
        element_1_num_pts = len(element_1_x_vals)
        element_2_num_pts = len(element_2_x_vals)

        #integrand_c0 = N_1_i_min1_x[element_1_num_pts:] * test_func_deriv[:element_1_num_pts]
        #integrand_c1 = N_2_i_min1_x[:element_1_num_pts] * test_func_deriv[:element_1_num_pts]
        #integrand_c2 = N_1_i_x[element_2_num_pts:] * test_func_deriv[element_2_num_pts:]
        #integrand_c3 = N_2_i_x[:element_2_num_pts] * test_func_deriv[element_2_num_pts:]

        print("reference node", node_list[x_nodes//2 - 1])

        integrand_c0 = N_1_i_min1_x[element_1_num_pts:] * (1)/delta_h **2
        integrand_c1 = N_2_i_min1_x[:element_1_num_pts] * (1)/delta_h **2
        integrand_c2 = N_1_i_x[element_2_num_pts:] * (-1)/delta_h **2
        integrand_c3 = N_2_i_x[:element_2_num_pts] * (-1)/delta_h **2

        integrand_b0 = local_x_distribution * test_func

        #print(element_1_x_vals[0], element_1_x_vals[-1])

        c0 = simpson_integrate(element_1_x_vals, integrand_c0, element_1_num_pts)
        c1 = simpson_integrate(element_1_x_vals, integrand_c1, element_1_num_pts)
        c2 = simpson_integrate(element_2_x_vals, integrand_c2, element_2_num_pts)
        c3 = simpson_integrate(element_2_x_vals, integrand_c3, element_2_num_pts)
    
        b0 = simpson_integrate(local_x_distribution, integrand_b0, nodes_per_subdomain)
        
        A_matrix[i, i - 1] = -c0 
                        
        A_matrix[i, i] = -(c1 + c2)

        A_matrix[i, i + 1] = -c3 

        print("c0", c0, "c1", c1 , "c2", c2, "c3", c3)

        for k in range(0, x_nodes):

            if k != i and k != i-1 and k != i+1:

                A_matrix[i, k] = 0

        b_matrix[i] = -b0 

A_matrix_with_BC, b_matrix_with_BC = Apply_BC2(A_matrix, b_matrix)    

print("System of Equations:")
print("", A_matrix_with_BC[0], "|", b_matrix_with_BC[0])
print("", A_matrix_with_BC[1], "|", b_matrix_with_BC[1])
print("", A_matrix_with_BC[2], "|", b_matrix_with_BC[2])
print("", A_matrix_with_BC[3], "|", b_matrix_with_BC[3])
print("", A_matrix_with_BC[4], "|", b_matrix_with_BC[4])
print("", A_matrix_with_BC[5], "|", b_matrix_with_BC[5])

#continous true solution
x_true = x_distribution
y_true = -x_true**3/6 +3/2*x_true

u_soln = sp.linalg.solve(A_matrix_with_BC, b_matrix_with_BC)

u_string = []
for i in range(len(u_soln)):

    u_string.append(str(u_soln[i]))

u_soln_string = ",".join(u_string)

file_path = r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\Training data\FEM Displacement data.txt"

file = open(file_path, "w")
file.write(u_soln_string)

#Residual Calculation

residual = 0

for i in range(1, len(x_true)):

    residual += 1/(len(x_true)) * abs((u_soln[i] - y_true[i])/y_true[i])

print("L2 Residual:", residual)

plt.grid()
plt.title("15 node FEM solution for continous bar")
plt.xlabel("x")
plt.ylabel("u")
plt.plot(x_true, y_true, color = "orange", label = "True Solution")
plt.plot(x_distribution, u_soln, color = "b", label = "FEM solution")
plt.legend()

plt.show()