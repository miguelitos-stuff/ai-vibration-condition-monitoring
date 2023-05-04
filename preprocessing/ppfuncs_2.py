# Import libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io


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
            sample_val = sensor_data[(i * sample_size):((i + 1) * sample_size)]
            sample_val = sample_val[None, :]
            sample_time = torch.arange(0, (sample_size / 40000), 1/40000).to(device)
            sample_time = sample_time[None, :]
            sample_ = torch.cat((sample_val, sample_time), 0)
            sample_ = sample_[None, :]
            samples_ = torch.cat((samples_, sample_), 0)
    return samples_


def plot_spectrogram_2():
    pass


def plot_spectrogram():
    # Define the list of frequencies
    frequencies = np.arange(5, 105, 5)
    sampling_frequency = 400

    s1 = np.empty([0])  # For samples
    s2 = np.empty([0])  # For signal

    # Start and Stop Value of the sample
    start = 1
    stop = sampling_frequency + 1

    for frequency in frequencies:
        sub1 = np.arange(start, stop, 1)
        # Signal - Sine wave with varying frequency + Noise
        sub2 = np.sin(2 * np.pi * sub1 * frequency * 1 / sampling_frequency) + np.random.randn(len(sub1))
        s1 = np.append(s1, sub1)
        s2 = np.append(s2, sub2)
        start = stop + 1
        stop = start + sampling_frequency

    # Plot the signal
    plt.subplot(211)
    plt.plot(s1, s2)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    # Plot the spectrogram
    plt.subplot(212)
    power_spectrum, frequencies_found, time, image_axis = plt.specgram(s2, Fs=sampling_frequency)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()


if __name__ == '__main__':
    # test your functions here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "data/Healthy/H"
    sensor_data = extract_data_2(path, '2', 4, device=device)
    sensor_samples = create_samples(sensor_data, 5, 40000, device=device)
    print(sensor_samples)
    # plot_spectrogram()
