# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import torch
from scipy import fftpack
from ppfuncs import extract_data_2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Code will be executed on:", device)

#Extracting data
path = "data/Healthy/H"
time_start = 1
time_end = 10
sen_start = 3
sen_end = 10


data = torch.zeros()
for i in range(time_start, time_end + 1):
    for j in range(sen_start, sen_end+1):
        data.append(extract_data_2(path, i, j, device=device))


def amplitude_frequency_plotting(data):

    sampling_rate = 40000 #[Hz]
    time = 10*60*8 #is this just 10 minutes times 8 sensors
    n_samples = sampling_rate * time

    data_fft = scipy.fftpack.fft(data)
    data_amp = 2 / n_samples * np.abs(data_fft)
    data_freq = np.abs(scipy.fftpack.fftfreq(n_samples, 1/40000))


    #Plot amplitude vs. frequency
    plt.plot(data_freq, data_amp)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.show()

    #Make images with pixels representing the magnitude at a certain frequency

    amplitude_norm = normalize(data_amp)
    images = np.reshape(amplitude_norm, (224,224))

    return images

if __name__ == '__main__':
    amplitude_frequency_plotting(data)