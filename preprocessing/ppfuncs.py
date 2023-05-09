# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import torch
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


def extract_data(path, n, s):
    mat_file = scipy.io.loadmat(path+f'{n}.mat')
    data_np = mat_file[f'AN{s}']
    data_np = data_np.flatten()
    data_tensor = torch.tensor(data_np)
    return data_tensor


def normalize(a_, end=255):
    # Normalizing so al values are between 0 and end value
    min_ = a_.min()
    a_ = a_ - min_
    max_ = a_.max()
    a_ = a_ * end / max_
    return a_


def create_matrix(vec):
    # Create a normalized matrix of the sample vector
    n = math.floor(vec.size(dim=0) ** 0.5)
    m = n
    rest = vec.size(dim=0) - n * m
    if rest != 0:
        vec_rest = torch.split(vec, [n*m, rest])
        vec = vec_rest[0]
    vec = normalize(vec)
    matrix = torch.reshape(vec, (1, n, m))
    return matrix


def generate_samples(sensor_data, n_samples, sample_size, spacing=False):
    # input is one minute of datapoints of one sensor [tensor]
    samples_ = torch.Tensor([])
    if spacing:
        rest = sensor_data.shape[0] - n_samples * sample_size ** 2
        space = rest // n_samples
        b = 0
        e = sample_size**2
        for i in range(n_samples):
            samples_ = torch.cat((samples_, sensor_data[b:e]), 0)
            b = e + space
            e += (sample_size**2 + space)
    else:
        for i in range(n_samples):
            sample_ = sensor_data[(i * sample_size ** 2):((i + 1) * sample_size ** 2)]
            sample_ = sample_[None, :]
            samples_ = torch.cat((samples_, sample_), 0)
    samples_2 = torch.Tensor([])
    for j in range(n_samples):
        sample_ = create_matrix(samples_[j])
        samples_2 = torch.cat((samples_2, sample_), 0)
    return samples_2


def visualize(data_matrix):
    # Take the normalized data matrices and visualize
    plt.imshow(data_matrix, cmap='viridis')
    plt.colorbar()
    plt.show()
    return


def visualize_compare(data_healthy, data_damaged, n):
    fig, axs = plt.subplots(2, n)
    n_2 = int(n/2-0.4)
    axs[0, n_2].set_title('Healthy')
    axs[1, n_2].set_title('Damaged')
    for i in range(0, n):
        axs[0, i].imshow(data_healthy[i], cmap='viridis')
        axs[1, i].imshow(data_damaged[i], cmap='viridis')
    plt.setp(axs, xticks=[], yticks=[])
    plt.show()
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
    return train_data_dict_, val_data_dict_, test_data_dict_


if __name__ == '__main__':
    # test your functions here

    # Visualize the data matrix
    test_matrix = np.random.randint(low=0, high=256, size=(224, 224))
    test_matrix = torch.Tensor(test_matrix)
    test_matrix = test_matrix.numpy()
    print(test_matrix)

    torch.Tensor.ndim = property(lambda self: len(self.shape))
    visualize(test_matrix)

    # test sample making
    # a = torch.randint(0, 10, (2400000,))
    # samples = generate_samples(a, 20, 244, device='cpu')
    # print(samples)
    # print(samples.shape)

    # calculating average
    # device = "cpu"
    # path = "../data/Healthy/H"
    # data = extract_data_2(path, 1, 3, device=device)
    # images = generate_samples(data, 20, 244, device=device)
    # test_matrix = images[1].numpy()
    # print(np.average(test_matrix))
    # visualize(test_matrix)
    # test_matrix = test_matrix.flatten()
    # new_matrix = np.array([])
    # for i in range(len(test_matrix)):
    #     if i / 2 == 0:
    #         new_matrix = np.append(new_matrix, test_matrix[i])
    # print(np.average(new_matrix))
