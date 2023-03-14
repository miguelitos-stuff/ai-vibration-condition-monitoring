#import numpy as np
import matplotlib.pyplot as plt
import scipy.io

#Make sure to put the file in the same directory as the pycharm project so the line below can operate smoothly.
x = scipy.io.loadmat('Damaged/D1.mat')
#print(x)


AN3_list = [[element for element in upperElement] for upperElement in x['AN3']]
print(AN3_list)
plt.figure()
plt.plot(AN3_list)
plt.show()