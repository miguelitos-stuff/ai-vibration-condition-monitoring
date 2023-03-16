#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
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


#Normalizing the data

def create_matrix():
    pass

def normalize():
    pass



def generate_sample():
    pass


# Visualizing the normlaized data
def visualize(data_matrix):
    # Take the normalized data matrices (that have entries with values ranging from 0 to 255) and visualize these on a grey scale

    plt.imshow(data_matrix, cmap='viridis')
    plt.colorbar()
    plt.show()
    return


