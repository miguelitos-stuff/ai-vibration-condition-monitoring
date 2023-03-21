# Import libraries
import matplotlib.pyplot as plt
import scipy.io
import numpy as np

# Make sure to put the file in the same directory as the pycharm project so the line below can operate smoothly.
sensor_mat = scipy.io.loadmat('data/Damaged/D1.mat')
x = np.linspace(0, 60, 40000*60)
# Choose the time interval you want to see and which sensor
# time interval: 0 - 60 (seconds)
# Sensors: AN3 to AN10
t_begin = 1
t_end = t_begin
sensor = 'AN4'

tb = int(t_begin * 40000)
te = int(t_end * 40000) + 60025

y = np.array(sensor_mat[sensor])
y = y.flatten()
print(y)
print(sensor_mat)
print(len(y))
plt.figure()
plt.plot(x[tb:te], y[tb:te])
plt.show()
