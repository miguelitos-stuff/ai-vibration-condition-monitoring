# Main file loop for preprocessin
import matplotlib.pyplot as plt
from ppfuncs import visualize
import numpy as np


def generate_sample():
    pass

def normalize():
    pass

def visualize(matrix):
    #Visualize the data matrix
    test_matrix = np.random.randint(low=0, high=256, size=(224, 224))
    visualize(test_matrix)


if __name__ == '__main__':
    #put your code to test the function in here
