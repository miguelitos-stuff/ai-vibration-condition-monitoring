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

    print_ = True

    # Define import variables for both Healty and Damaged
    time_start = 1
    time_end = 10
    sen_start = 3
    sen_end = 10

    # Define image variables for both Healty and Damaged
    n_images_sensor = 20
    images_size = 224

    # Create list of healthy images, by extracting data and creating images with above defined variables
    healthy_image_list = torch.Tensor([])
    healthy_image_list.to("cuda")
    path = "data/Healthy/H"
    if print_: print("Start on healthy data")
    for i in range(time_start, time_end+1):
        for j in range(sen_start, sen_end+1):
            data = pp.extract_data_2(path, i, j)
            images = pp.generate_samples(data, n_images_sensor, images_size)
            healthy_image_list = torch.cat((healthy_image_list, images), 0)
        if print_: print(f"\tHealthy data at time {i}: Completed")
    if print_: print("Healthy data: Completed \n\nStart on Damaged data")

    # Create list of damaged images, by extracting data and creating images with above defined variables
    damaged_image_list = torch.Tensor([])
    damaged_image_list.to("cuda")
    path = "data/Damaged/D"
    for i in range(time_start, time_end+1):
        for j in range(sen_start, sen_end+1):
            data = pp.extract_data_2(path, i, j)
            images = pp.generate_samples(data, n_images_sensor, images_size)
            damaged_image_list = torch.cat((damaged_image_list, images), 0)
        if print_: print(f"\tDamaged data at time {i}: Completed")
    if print_: print("Damaged data: Completed \n\nStart on saving data")




    # pp.visualize_compare(healthy_image_list.numpy(), damaged_image_list.numpy(), 6)

    pp.save(healthy_image_list, damaged_image_list)
    if print_: print("Saving data: Completed")

