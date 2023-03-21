#Main file loop for preprocessin

#Import libraries
import matplotlib.pyplot as plt
from ppfuncs import visualize
#from ppfuncs import extract_data
import numpy as np
from pathlib import Path

#Visualize the data matrix
test_matrix = np.random.randint(low=0, high=256, size=(224, 224))
visualize(test_matrix)

#Make an array with all of the file numbers
datapath = Path('../data/')
filelist = []
for i in datapath.iterdir():
    for datafile in i.iterdir():
        filelist.append(datafile)
ordered_list = sorted(filelist, key = lambda x: x.stem[-6:-5])
print(filelist)

