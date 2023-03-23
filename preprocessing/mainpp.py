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


# #Make an array with all of the file numbers
# datapath = Path('../data/')
# filelist = []
# for i in datapath.iterdir():
#     for datafile in i.iterdir():
#         filelist.append(datafile)
# ordered_list = sorted(filelist, key = lambda x: x.stem[-6:-5])
# print(filelist)

if __name__ == '__main__':

    # Define import variables for both Healty and Damaged
    time_start = 6
    time_end = 6
    sen_start = 3
    sen_end = 3

    # Define image variables for both Healty and Damaged
    n_images_sensor = 20
    images_size = 224

    # Create list of healthy images, by extracting data and creating images with above defined variables
    healthy_image_list = torch.Tensor([])
    path = "data/Healthy/H"
    for i in range(time_start, time_end+1):
        for j in range(sen_start, sen_end+1):
            data = pp.extract_data_2(path, i, j)
            images = pp.generate_samples(data, n_images_sensor, images_size)
            healthy_image_list = torch.cat((healthy_image_list, images), 0)

    # Create list of damaged images, by extracting data and creating images with above defined variables
    damaged_image_list = torch.Tensor([])
    path = "data/Damaged/D"
    for i in range(time_start, time_end+1):
        for j in range(sen_start, sen_end+1):
            data = pp.extract_data_2(path, i, j)
            images = pp.generate_samples(data, n_images_sensor, images_size)
            damaged_image_list = torch.cat((damaged_image_list, images), 0)




    # pp.visualize_compare(healthy_image_list.numpy(), damaged_image_list.numpy(), 8)

    pp.save(healthy_image_list, damaged_image_list)

