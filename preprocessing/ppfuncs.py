#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torch

#Extracting the data
def extract_data(path, AN_beginning, AN_end, amount_of_data_points):




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


