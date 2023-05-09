# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import torch
from scipy import fftpack
from ppfuncs import normalize

def extract_data_2(path, n, s, device):
    mat_file = scipy.io.loadmat(path+f'{n}.mat')
    data_np = mat_file[f'AN{s}']
    data_np = data_np.flatten()
    data_tensor = torch.tensor(data_np).to(device)
    return data_tensor

def create_samples(sensor_data, n_samples, sample_size, device, spacing=False):
    # create samples with time and frequency arrays
    samples_ = torch.Tensor([]).to(device)
    if spacing:
        pass
    else:
        for i in range(n_samples):
            sample_ = sensor_data[(i * sample_size):((i + 1) * sample_size)]
            sample_ = sample_[None, :]
            samples_ = torch.cat((samples_, sample_), 0)
    return samples_

def amplitude_frequency_plotting(samples_):
    sample_ = samples_.detach().cpu().numpy()
    sampling_rate = 40000 #[Hz]
    time = 1*60
    n_samples = sampling_rate * time

    data_fft = scipy.fftpack.fft(sample_)
    data_amp = 2 / n_samples * np.abs(data_fft)
    data_freq = np.abs(scipy.fftpack.fftfreq(n_samples, 1/40000))


    #Plot amplitude vs. frequency
    # plt.plot(data_freq, data_amp)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')
    # plt.show()

    #Make images with pixels representing the magnitude at a certain frequency

    amplitude_norm = normalize(data_amp)
    images = np.reshape(amplitude_norm, (240,10000))

    return images

def visualize(image):
    # Take the normalized data matrices and visualize
    plt.imshow(image, cmap='viridis')
    plt.colorbar()
    plt.show()
    return


if __name__ == '__main__':
    #Execute code on cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Code will be executed on:", device)

    # Extracting data
    sensor_data_damaged = extract_data_2("data/Damaged/D", '2', 4, device=device)
    sensor_data_healthy = extract_data_2("data/Healthy/H", '2', 4, device=device)

    sensor_samples_damaged = create_samples(sensor_data_damaged, 10, 40000, device=device)
    sensor_samples_healthy = create_samples(sensor_data_healthy, 10, 40000, device=device)

    #Create amplitude_frequency_images
    Damaged_images = amplitude_frequency_plotting(sensor_data_damaged)
    Healthy_images = amplitude_frequency_plotting(sensor_data_healthy)

    #Visualize
    visualize(Damaged_images)
    print(Damaged_images)
