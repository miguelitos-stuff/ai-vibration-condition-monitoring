# Import libraries
import matplotlib.pyplot as plt
import scipy.io
import numpy as np


# Make sure to put the file in the same directory as the pycharm project so the line below can operate smoothly.
sensor_mat = scipy.io.loadmat('preprocessing/data/Healthy/H1.mat')
x = np.linspace(0, 60, 40000*60)
# Choose the time interval you want to see and which sensor
# time interval: 0 - 60 (seconds)
# Sensors: AN3 to AN10
t_begin = 1
t_end = t_begin
sensor = ['AN3', 'AN4', 'AN5', 'AN6', 'AN7', 'AN8', 'AN9', 'AN10']

tb = int(t_begin * 40000)
te = int(t_end * 40000) + int(100)

plt.figure()
for i in range(8):
    a = 7
    if i == 6 or i == 5: a = 15
    y = sensor_mat[sensor[i]] + i*a
    y = y.flatten()
    plt.plot(x[tb:te], y[tb:te], label=sensor[i])
plt.legend()
plt.show()


