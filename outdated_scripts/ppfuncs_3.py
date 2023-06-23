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
    print(data_tensor.size())
    return data_tensor


def create_samples(sensor_data, n_samples, sample_size, device, spacing=False):
    # create samples with time and frequency arrays
    # returns amplitude vs time, with time 1 minute and delta t = 1/40000
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
    # sampling_rate in [Hz]
    samples_ = samples_.detach().cpu().numpy()
    sampling_rate = 40000
    time = 1*60
    n_samples = sampling_rate * time

    # data_fft = scipy.fftpack.fft(sample_)
    # data_amp = 2 / n_samples * np.abs(data_fft)
    # data_freq = np.abs(scipy.fftpack.fftfreq(n_samples, 1/40000))

    data_amplitude = np.abs(np.fft.fft(samples_))
    data_freqs = np.fft.fftfreq(len(data_amplitude), d=1.0 / sampling_rate)

    # Plot amplitude vs. frequency
    plt.plot(data_freqs, data_amplitude)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.show()

    # Make images with pixels representing the magnitude at a certain frequency
    amplitude_norm = normalize(data_amplitude)
    images = np.reshape(amplitude_norm, (200,200))
    return images


def visualize(image):
    # Take the normalized data matrices and visualize
    plt.imshow(image, cmap='viridis')
    plt.colorbar()
    plt.savefig("preprocessing")
    return
