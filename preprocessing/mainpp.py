#Main file loop for preprocessin

#Import libraries
import matplotlib.pyplot as plt
import torch
from ppfuncs import visualize
from ppfuncs import extract_data
import ppfuncs as pp
import numpy as np
import scipy.io

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



# Define import variables for both Healty and Damaged
time_start = 1
time_end = 10
sen_start = 3
sen_end = 10

# Define image variables for both Healty and Damaged
n_images_sensor = 20
images_size = 224

# Create list of healthy images, by extracting data and creating images with above defined variables
healthy_image_list = torch.Tensor([0])
path = "data/Healthy/H"
for i in range(time_start, time_end+1):
    for j in range(sen_start, sen_end+1):
        data = pp.extract_data_2(path, i, j)
        images = pp.generate_samples(data, n_images_sensor, images_size)
        healthy_image_list = pp.tensor_append(healthy_image_list, images)

# Create list of damaged images, by extracting data and creating images with above defined variables
damaged_image_list = torch.Tensor([0])
path = "data/Damaged/D"
for i in range(time_start, time_end+1):
    for j in range(sen_start, sen_end+1):
        data = pp.extract_data_2(path, i, j)
        images = pp.generate_samples(data, n_images_sensor, images_size)
        damaged_image_list = pp.tensor_append(damaged_image_list, images)

