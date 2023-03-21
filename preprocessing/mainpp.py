#Main file loop for preprocessing

#Import libraries
import matplotlib.pyplot as plt
import numpy as np

from ppfuncs import extract_data
from ppfuncs import normalize
from ppfuncs import create_matrix
from ppfuncs import generate_sample
from ppfuncs import visualize


if __name__ == "__main__":

    raw_data =

    matrix = create_matrix(raw_data)

    normalized_data = normalize(matrix)

    plot = visualize(normalized_data)


    return normalized_data, plot




# #Visualize the data matrix
# test_matrix = np.random.randint(low=0, high=256, size=(224, 224))
# visualize(test_matrix)

