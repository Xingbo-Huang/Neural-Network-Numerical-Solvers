import numpy as np
import matplotlib.pyplot as plt

def f(x, t):

    return np.power(x, 1)

def g(x, t):

    return -np.square(x)

def RK4(t0, tf, x0, h):

    x_list = []
    x_dot_list = []

    t0 = t0
    x0 = x0
    x_dot0 = f(x0, t0)

    for i in range(abs(int((tf-t0)//h))):

        K1 = h * f(x0, t0)
        K2 = h * f(x0 + K1/2, t0 + h/2)
        K3 = h * f(x0 + K2/2, t0 +h/2) 
        K4 = h * f(x0 + K3, t0 + h)

        x1 = x0 + 1/6 * (K1 + 2 * K2 + 2 * K3 + K4)

        t0 = t0 + h
        x0 = x1

        x_dot1 = f(x1, t0)

        x_list.append(x0)
        x_dot_list.append(x_dot1)

    return x_list, x_dot_list

steps = 49

t0 = 1
tf = 2.0
z_t0 = 1
step_size = (tf-t0)/steps

x_list, x_dot_list = RK4(t0, tf, z_t0, step_size)

print(len(x_list))
print(len(x_dot_list))

time_list = np.linspace(t0, tf, steps)

#File making ----------------------------------------------------

x_string = []
for i in range(len(x_list)):

    x_string.append(str(x_list[i]))

x_soln_string = ",".join(x_string)

file_path = r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\Training data\Dynamical Systems\x data"

file = open(file_path, "w")
file.write(x_soln_string)

x_dot_string = []
for i in range(len(x_dot_list)):

    x_dot_string.append(str(x_dot_list[i]))

x_dot_soln_string = ",".join(x_dot_string)

file_path = r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\Training data\Dynamical Systems\ x dot data"

file = open(file_path, "w")
file.write(x_dot_soln_string)

#-------------------------------------------------------------

print(x_list)
print(x_dot_list)

plt.grid()
plt.plot(time_list, x_list, label = "x", color = "blue")
plt.plot(time_list, x_dot_list, label = "x dot", color = "red")
plt.legend()
plt.show()