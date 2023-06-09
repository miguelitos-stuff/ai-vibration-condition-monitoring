# Import libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader


class CreateDataset(Dataset):
    def __init__(self, label_tens, img_tens):
        self.img_labels = label_tens
        self.img_tens = img_tens.float()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_idx = int(self.img_labels[idx][0])
        image = self.img_tens[img_idx]
        label = self.img_labels[idx][1]
        return image, label


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


def spectrogram_2(sample_val_array, plot=False):
    # converts a sample into a spectrogram of 129 by x
    # x is dependent on the sample size
    sample_ = sample_val_array.detach().cpu().numpy()
    sampling_frequency = 40000
    spectrum, freq, time, fig = plt.specgram(sample_, Fs=sampling_frequency)
    spectrum = np.log10(spectrum)
    if plot:
        spec_plot = plt.imshow(spectrum)
        clb = plt.colorbar()
        plt.title('Damaged')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        clb.ax.set_title('[dB]')
        plt.show()
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


def create_data_dict(zeros_list, ones_list):
    labels_0 = torch.zeros(len(zeros_list))
    labels_1 = torch.ones(len(ones_list))
    labels_list = torch.cat((labels_1, labels_0), 0)
    labels_ind = torch.arange(0, len(labels_list))
    labels = torch.stack((labels_ind, labels_list), -1)
    images = torch.cat((zeros_list, ones_list), 0)
    data_dict_ = {"data": images[:, None, :, :], "label": labels}
    return data_dict_


def set_to_dict(data_, num_, ):
    dataloader = DataLoader(data_, batch_size=num_, shuffle=True)
    features, labels = next(iter(dataloader))
    ind = torch.arange(0, len(features))
    labels = torch.stack((ind, labels), -1)
    data_dict_ = {"data": features, "label": labels}
    return data_dict_


def split_data_dict(data_dict_, train_split_, val_split_, test_split_):
    all_data = CreateDataset(data_dict_["label"], data_dict_["data"])
    num_train = int(round(len(all_data) * train_split_, 0))
    num_val = int(round(len(all_data) * val_split_, 0))
    num_test = int(round(len(all_data) * test_split_, 0))
    (train_data_, val_data_, test_data_) = random_split(all_data, [num_train, num_val, num_test],
                                                        generator=torch.Generator().manual_seed(42))
    train_data_dict_ = set_to_dict(train_data_, num_train)
    val_data_dict_ = set_to_dict(val_data_, num_val)
    test_data_dict_ = set_to_dict(test_data_, num_test)
    return [train_data_dict_, val_data_dict_, test_data_dict_]
