#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import torch
import tensorflow as tf
import os

#Extracting the data
def extract_data(Damaged_or_Healthy, AN_beginning, AN_end, D_what):
    biggie_T = torch.empty(10)
    path_to_data_file = os.path.join(os.path.dirname(__file__), '..', 'data', str(Damaged_or_Healthy), 'D' + str(D_what) + ".mat")
    rawr = np.array(scipy.io.loadmat(path_to_data_file))
    print(rawr)
    for i in range(AN_beginning, AN_end+1):
        meow = tf.convert_to_tensor(rawr[i])
        meow = meow.flatten()
        biggie_T[i-AN_beginning] = meow
    return biggie_T

print(extract_data('Damaged',3,10,3))


def normalize(a, end=255, intit=True):
    # Normalizing so al values are between 0 and end value
    min = a.min()
    a = a - min
    max = a.max()
    a = a * end / max
    if intit:
        a = a.type(torch.int16)
    return a


def create_matrix(a):
    # Create a normalized matrix of the sample vector
    n = math.floor(a.size(dim=0) ** 0.5)
    m = n
    rest = a.size(dim=0) - n * m
    if (rest != 0):
        a_rest = torch.split(a, (n*m, rest))
        a = a_rest[0]
    a = normalize(a)
    matrix = torch.reshape(a, (n, m))
    return matrix




def generate_sample():
    pass


# Visualizing the normlaized data
def visualize(data_matrix):
    # Take the normalized data matrices (that have entries with values ranging from 0 to 255) and visualize these on a grey scale

    plt.imshow(data_matrix, cmap='viridis')
    plt.colorbar()
    plt.show()
    return


