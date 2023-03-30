# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import torch


def extract_data_2(path, n, s, device):
    mat_file = scipy.io.loadmat(path+f'{n}.mat')
    data_np = mat_file[f'AN{s}']
    data_np = data_np.flatten()
    data_tensor = torch.tensor(data_np).to(device)
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


def generate_samples(sensor_data, n_samples, sample_size, device, spacing=False):
    # input is one minute of datapoints of one sensor [tensor]
    samples_ = torch.Tensor([]).to(device)
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
    samples_2 = torch.Tensor([]).to(device)
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


def save(zeros_list, ones_list, device):
    labels_0 = torch.zeros(len(zeros_list)).to(device)
    labels_1 = torch.ones(len(ones_list)).to(device)
    labels = torch.cat((labels_1, labels_0), 0)
    images = torch.cat((zeros_list, ones_list), 0)
    data_dict = {"data": images, "label": labels}
    torch.save(data_dict, "data_dict.pt")
    return


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
