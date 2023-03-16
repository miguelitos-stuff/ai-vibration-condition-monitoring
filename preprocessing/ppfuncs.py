#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import math
import torch


#Extracting the data


def normalize(a, end = 255):
    # Normalizing so al values are between 0 and end value
    min = a.min()
    a = a + -1 * min
    max = a.max()
    a = a * end / max
    return a


def create_matrix(vector):
    # Create a normalized matrix of the sample vector

    vector = normalize(vector)

    a = torch.Tensor(np_a)
    n = math.floor(a.size(dim=0) ** 0.5)
    m = n
    rest = a.size(dim=0) - n * m
    if (rest != 0):
        a_rest = torch.split(a, (n * m, rest))
        a = a_rest[0]
    matrix = torch.reshape(a, (n, m))

    return matrix





def generate_sample():
    pass



#Making the matrices
def visualize(matrix):
    #Visualize the data matrix
    test_matrix = np.random.randint(low=0, high=256, size=(224, 224))
    visualize(test_matrix)




#Visualizing the normlaized data
#Take the normalized data matrices (that have entries with values ranging from 0 to 255) and visualize these on a grey scale
def visualize(data_matrix):
    plt.imshow(data_matrix)
    plt.show()
    return


