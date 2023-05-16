# Import libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io


def extract_data_2(path, n, s, device):
    # extracts data from the <.mat> files
    mat_file = scipy.io.loadmat(path+f'{n}.mat')
    data_np = mat_file[f'AN{s}']
    data_np = data_np.flatten()
    data_tensor = torch.tensor(data_np).to(device)
    return data_tensor


def create_samples(sensor_data, n_samples, sample_size, device):
    # create samples with a certain number of datapoints of the signal
    samples_ = torch.Tensor([]).to(device)
    for i in range(n_samples):
        sample_ = sensor_data[(i * sample_size):((i + 1) * sample_size)]
        sample_ = sample_[None, :]
        samples_ = torch.cat((samples_, sample_), 0)
    return samples_


def create_spectrograms(samples_, device):
    spectrograms_ = torch.Tensor([])
    for i in range(len(samples_)):
        spectrogram_ = spectrogram_2(samples_[i])
        spectrogram_ = torch.from_numpy(spectrogram_)
        spectrogram_ = spectrogram_[None, :]
        spectrograms_ = torch.cat((spectrograms_, spectrogram_), 0)
    spectrograms_ = spectrograms_.to(device)
    return spectrograms_


def generate_spectrogram_samples(sensor_data, n_samples, sample_size, device):
    samples_ = create_samples(sensor_data, n_samples, sample_size, device)
    spectrograms_tensor = create_spectrograms(samples_, device)
    return spectrograms_tensor


def spectrogram_2(sample_val_array):
    # converts a sample into a spectrogram of 129 by 129
    # note the second 129 is actually dependent on the sample size
    sample_ = sample_val_array.detach().cpu().numpy()
    sampling_frequency = 40000
    spectrum, freq, time, fig = plt.specgram(sample_, Fs=sampling_frequency)
    spectrum = np.log10(spectrum)
    # plot = plt.imshow(spectrum)
    # plt.colorbar()
    # plt.xlabel('Time')
    # plt.ylabel('Frequency')
    # plt.show()
    return spectrum


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


def save2(zeros_list, ones_list, device):
    labels_0 = torch.zeros(len(zeros_list)).to(device)
    labels_1 = torch.ones(len(ones_list)).to(device)
    labels = torch.cat((labels_1, labels_0), 0)
    images = torch.cat((zeros_list, ones_list), 0)
    data_dict = {"data": images, "label": labels}
    torch.save(data_dict, "data_dict_2.pt")
    return


if __name__ == '__main__':
    pass
    # test your functions here
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # sensor_data_damaged = extract_data_2("data/Damaged/D", '2', 9, device=device)
    # sensor_data_healthy = extract_data_2("data/Healthy/H", '2', 9, device=device)
    # sensor_samples_damaged = create_samples(sensor_data_damaged, 10, 16700, device=device)
    # sensor_samples_healthy = create_samples(sensor_data_healthy, 10, 16700, device=device)
    # spectrogram = spectrogram_2(sensor_samples_healthy[0])
    # print(type(spectrogram))
    # visualize_compare(sensor_samples_healthy, sensor_samples_damaged, 5)
