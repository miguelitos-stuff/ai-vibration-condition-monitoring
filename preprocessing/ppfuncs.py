#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import torch

#Extracting the data
def extract_data(path, AN_beginning, AN_end, amount_of_data_points):
    tensors = torch.empty(0,0)

    for i in range(AN_beginning,AN_end+1):
        mat_file = scipy.io.loadmat(path+str(i)+'.mat')
        for n in range(AN_beginning,AN_end+1):
            numpy_array = mat_file['AN'+str(n)]
            numpy_array = numpy_array.flatten()
            tensors.append(numpy_array)
    return tensors



print(extract_data('data/Damaged/D', 3,10,10))


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


