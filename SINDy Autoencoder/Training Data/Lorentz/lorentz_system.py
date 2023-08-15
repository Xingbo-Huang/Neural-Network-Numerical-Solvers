import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def RHS_z1(z1, z2, z3, t):

    return -10 * z1 + 10 * z2

def RHS_z2(z1, z2, z3, t):

    return 28* z1 - z2 - z1 * z3

def RHS_z3(z1, z2, z3, t):

    return -8/3 * z3 + z1 * z2

def RK4(t0, tf, z1_0, z2_0, z3_0, h):

    z1_list = []
    z2_list = []
    z3_list = []
    
    z1_dot_list = []
    z2_dot_list = []
    z3_dot_list = []

    t0 = t0

    z1_0 = z1_0
    z2_0 = z2_0
    z3_0 = z3_0

    z1_dot0 = RHS_z1(z1_0, z2_0, z3_0, t0)
    z1_dot0 = RHS_z2(z1_0, z2_0, z3_0, t0)
    z1_dot0 = RHS_z3(z1_0, z2_0, z3_0, t0)

    for i in range(abs(int((tf-t0)//h))):

        K1_z1 = h * RHS_z1(z1_0, z2_0, z3_0, t0)
        K1_z2 = h * RHS_z2(z1_0, z2_0, z3_0, t0)
        K1_z3 = h * RHS_z3(z1_0, z2_0, z3_0, t0)

        K2_z1 = h * RHS_z1(z1_0 + K1_z1/2, z2_0 + K1_z2/2 , z3_0 + K1_z3/2 ,t0 + h/2)
        K2_z2 = h * RHS_z2(z1_0 + K1_z1/2, z2_0 + K1_z2/2 , z3_0 + K1_z3/2 ,t0 + h/2)
        K2_z3 = h * RHS_z3(z1_0 + K1_z1/2, z2_0 + K1_z2/2 , z3_0 + K1_z3/2 ,t0 + h/2)

        K3_z1 = h * RHS_z1(z1_0 + K2_z1/2, z2_0 + K2_z2/2 , z3_0 + K2_z3/2 ,t0 + h/2) 
        K3_z2 = h * RHS_z2(z1_0 + K2_z1/2, z2_0 + K2_z2/2 , z3_0 + K2_z3/2 ,t0 + h/2) 
        K3_z3 = h * RHS_z3(z1_0 + K2_z1/2, z2_0 + K2_z2/2 , z3_0 + K2_z3/2 ,t0 + h/2) 

        K4_z1 = h * RHS_z1(z1_0 + K3_z1, z2_0 + K3_z2 , z3_0 + K3_z3 ,t0 + h)
        K4_z2 = h * RHS_z2(z1_0 + K3_z1, z2_0 + K3_z2 , z3_0 + K3_z3 ,t0 + h)
        K4_z3 = h * RHS_z3(z1_0 + K3_z1, z2_0 + K3_z2 , z3_0 + K3_z3 ,t0 + h)

        z1_1 = z1_0 + 1/6 * (K1_z1 + 2 * K2_z1 + 2 * K3_z1 + K4_z1)
        z2_1 = z2_0 + 1/6 * (K1_z2 + 2 * K2_z2 + 2 * K3_z2 + K4_z2)
        z3_1 = z3_0 + 1/6 * (K1_z3 + 2 * K2_z3 + 2 * K3_z3 + K4_z3)

        t0 = t0 + h

        z1_0 = z1_1
        z2_0 = z2_1
        z3_0 = z3_1

        z1_dot_1 = RHS_z1(z1_0, z2_0, z3_0, t0)
        z2_dot_1 = RHS_z2(z1_0, z2_0, z3_0, t0)
        z3_dot_1 = RHS_z3(z1_0, z2_0, z3_0, t0)

        z1_list.append(z1_0)
        z2_list.append(z2_0)
        z3_list.append(z3_0)

        z1_dot_list.append(z1_dot_1)
        z2_dot_list.append(z2_dot_1)
        z3_dot_list.append(z3_dot_1)

    return z1_list, z2_list, z3_list, z1_dot_list, z2_dot_list, z3_dot_list

steps = 100

z1_list, z2_list, z3_list, z1_dot_list, z2_dot_list, z3_dot_list = RK4(0, 1, 0.1, 0, 0, 1/steps)

# print(len(x_list))
# print(len(x_dot_list))

time_list = np.linspace(0, np.pi, steps)

#File making ----------------------------------------------------

z1_string = []
for i in range(len(z1_list)):

    z1_string.append(str(z1_list[i]))

z1_soln_string = ",".join(z1_string)

file_path = r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\Training data\Dynamical Systems\Lorentz\x1 data"

file = open(file_path, "w")
file.write(z1_soln_string)

######

z2_string = []
for i in range(len(z2_list)):

    z2_string.append(str(z2_list[i]))

z2_soln_string = ",".join(z2_string)

file_path = r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\Training data\Dynamical Systems\Lorentz\x2 data"

file = open(file_path, "w")
file.write(z2_soln_string)

######

z3_string = []
for i in range(len(z3_list)):

    z3_string.append(str(z3_list[i]))

z3_soln_string = ",".join(z3_string)

file_path = r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\Training data\Dynamical Systems\Lorentz\x3 data"

file = open(file_path, "w")
file.write(z3_soln_string)

######

z1_dot_string = []
for i in range(len(z1_dot_list)):

    z1_dot_string.append(str(z1_dot_list[i]))

z1_dot_soln_string = ",".join(z1_dot_string)

file_path = r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\Training data\Dynamical Systems\Lorentz\x1 dot data"

file = open(file_path, "w")
file.write(z1_dot_soln_string)

######

z2_dot_string = []
for i in range(len(z2_dot_list)):

    z2_dot_string.append(str(z2_dot_list[i]))

z2_dot_soln_string = ",".join(z2_dot_string)

file_path = r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\Training data\Dynamical Systems\Lorentz\x2 dot data"

file = open(file_path, "w")
file.write(z2_dot_soln_string)

######

z3_dot_string = []
for i in range(len(z3_dot_list)):

    z3_dot_string.append(str(z3_dot_list[i]))

z3_dot_soln_string = ",".join(z3_dot_string)

file_path = r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\Training data\Dynamical Systems\Lorentz\x3 dot data"

file = open(file_path, "w")
file.write(z3_dot_soln_string)

#-------------------------------------------------------------

# print(x_list)
# print(x_dot_list)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# z1_list, z2_list = np.meshgrid(z1_list, z2_list)

print(len(z1_list))
print(len(z2_list))
print(len(z3_list))

ax.set_xlabel("z1")
ax.set_ylabel("z2")
ax.set_zlabel("z3")

ax.plot(z1_list, z2_list, z3_list, color = "blue", lw=0.5)
ax.grid()

plt.show()