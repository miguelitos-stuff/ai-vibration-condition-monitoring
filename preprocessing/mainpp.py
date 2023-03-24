# Main file loop for preprocessing

# # Import libraries
# import matplotlib.pyplot as plt
# import numpy as np
# import scipy.io
# from pathlib import Path

import torch
import ppfuncs as pp

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Code will be executed on:", device)


if __name__ == '__main__':
    # Set to False if you don't want to print the proces
    print_ = True

    # Define import variables for both Healthy and Damaged
    time_start = 1
    time_end = 10
    sen_start = 3
    sen_end = 10

    # Define image variables for both Healthy and Damaged
    n_images_sensor = 20
    images_size = 224

    # Create list of healthy images, by extracting data and creating images with above defined variables
    healthy_image_list = torch.Tensor([]).to(device)
    path = "data/Healthy/H"
    if print_:
        print("Start on healthy data")
    for i in range(time_start, time_end+1):
        for j in range(sen_start, sen_end+1):
            data = pp.extract_data_2(path, i, j, device=device)
            images = pp.generate_samples(data, n_images_sensor, images_size, device=device)
            healthy_image_list = torch.cat((healthy_image_list, images), 0)
        if print_:
            print(f"\tHealthy data at time {i}: Completed")
    if print_:
        print("Healthy data: Completed \n\nStart on Damaged data")

    # Create list of damaged images, by extracting data and creating images with above defined variables
    damaged_image_list = torch.Tensor([]).to(device)
    path = "data/Damaged/D"
    for i in range(time_start, time_end+1):
        for j in range(sen_start, sen_end+1):
            data = pp.extract_data_2(path, i, j, device=device)
            images = pp.generate_samples(data, n_images_sensor, images_size, device=device)
            damaged_image_list = torch.cat((damaged_image_list, images), 0)
        if print_:
            print(f"\tDamaged data at time {i}: Completed")
    if print_:
        print("Damaged data: Completed \n\nStart on saving data")

    # pp.visualize_compare(healthy_image_list.numpy(), damaged_image_list.numpy(), 6)

    pp.save(healthy_image_list, damaged_image_list, device=device)
    if print_:
        print("Saving data: Completed")
