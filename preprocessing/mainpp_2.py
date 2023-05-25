# Main file loop for preprocessing

# # Import libraries
# import matplotlib.pyplot as plt
# import numpy as np
# import scipy.io
# from pathlib import Path

import torch
import ppfuncs_2 as pp2

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
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
    max_images = False
    sample_size = 16700
    if max_images:
        count_datapoints = 60 * 40 * 1000
        n_samples = int(count_datapoints / sample_size)
    else:
        n_samples = 30

    print(n_samples)

    # Create list of healthy images, by extracting data and creating images with above defined variables
    healthy_spectrogram_tensor = torch.Tensor([]).to(device)

    path = "data/Healthy/H"
    if print_:
        print("Start on healthy data")
    for i in range(time_start, time_end+1):
        for j in range(sen_start, sen_end+1):
            data = pp2.extract_data_2(path, i, j, device=device)
            spectrograms = pp2.generate_spectrogram_samples(data, n_samples, sample_size, device=device)
            healthy_spectrogram_tensor = torch.cat((healthy_spectrogram_tensor, spectrograms), 0)
        if print_:
            print(f"\tHealthy data at time {i}: Completed")
    if print_:
        print("Healthy data: Completed \n\nStart on Damaged data")

    # Create list of damaged images, by extracting data and creating images with above defined variables
    damaged_spectrogram_tensor = torch.Tensor([]).to(device)
    path = "data/Damaged/D"
    for i in range(time_start, time_end+1):
        for j in range(sen_start, sen_end+1):
            data = pp2.extract_data_2(path, i, j, device=device)
            spectrograms = pp2.generate_spectrogram_samples(data, n_samples, sample_size, device=device)
            damaged_spectrogram_tensor = torch.cat((damaged_spectrogram_tensor, spectrograms), 0)
        if print_:
            print(f"\tDamaged data at time {i}: Completed")
    if print_:
        print("Damaged data: Completed \n\nStart on saving data")

    print(healthy_spectrogram_tensor.device)

    spectrogram_dict = pp2.create_data_dict(healthy_spectrogram_tensor, damaged_spectrogram_tensor)
    # torch.save(spectrogram_dict, "data_dict_2.pt")
    if print_:
        print("Saving data: Completed")

    tp = 0.65
    vp = 0.20
    testp = 0.15
    train_data_dict, val_data_dict, test_data_dict = pp2.split_data_dict(spectrogram_dict, tp, vp, testp)
    if print_:
        print("Splitting datasets: Completed")
    torch.save(train_data_dict, "../train_spec_dict.pt")
    torch.save(val_data_dict, "../val_spec_dict.pt")
    torch.save(test_data_dict, "../test_spec_dict.pt")

    if print_:
        print("Saving data: Completed")
