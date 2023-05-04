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
            sample_ = sensor_data[(i * sample_size):((i + 1) * sample_size)]
            sample_ = sample_[None, :]
            samples_ = torch.cat((samples_, sample_), 0)
    return samples_


def plot_spectrogram_2(sample_val_array):
    sample_ = sample_val_array.detach().cpu().numpy()
    sampling_frequency = 40000
    power_spectrum, frequencies_found, time, image_axis = plt.specgram(sample_, Fs=sampling_frequency)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()


def visualize_compare(data_healthy, data_damaged, n):
    samples_healthy = data_healthy.detach().cpu().numpy()
    samples_damaged = data_damaged.detach().cpu().numpy()
    sampling_frequency = 40000
    fig, axs = plt.subplots(2, n)
    n_2 = int(n/2-0.4)
    axs[0, n_2].set_title('Healthy')
    axs[1, n_2].set_title('Damaged')
    for i in range(0, n):
        power_spectrum_h, frequencies_found_h, time_h, image_axis_h = axs[0, i].specgram(samples_healthy[i], Fs=sampling_frequency)
        power_spectrum_d, frequencies_found_d, time_d, image_axis_d = axs[1, i].specgram(samples_damaged[i], Fs=sampling_frequency)
    plt.setp(axs, xticks=[], yticks=[])
    plt.show()
    return


if __name__ == '__main__':
    # test your functions here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sensor_data_damaged = extract_data_2("data/Damaged/D", '2', 4, device=device)
    sensor_data_healthy = extract_data_2("data/Healthy/H", '2', 4, device=device)
    sensor_samples_damaged = create_samples(sensor_data_damaged, 10, 40000, device=device)
    sensor_samples_healthy = create_samples(sensor_data_healthy, 10, 40000, device=device)
    visualize_compare(sensor_samples_healthy, sensor_samples_damaged, 5)
    # plot_spectrogram_2(sensor_samples_damaged[0])
    # plot_spectrogram()
