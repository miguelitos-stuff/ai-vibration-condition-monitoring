#Main file loop for preprocessin

#Import libraries
import matplotlib.pyplot as plt
from ppfuncs import visualize
from ppfuncs import extract_data
import numpy as np


#Visualize the data matrix
test_matrix = np.random.randint(low=0, high=256, size=(224, 224))
visualize(test_matrix)

