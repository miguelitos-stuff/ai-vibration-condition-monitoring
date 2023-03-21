#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import torch
import os

#Extracting the data
def extract_data(Damaged_or_Healthy, AN_beginning, AN_end, D_what):
    biggie_T = torch.empty(10)
    print(biggie_T)
    path_to_data_file = os.path.join(os.path.dirname(__file__), '..', 'data', str(Damaged_or_Healthy), 'D' + str(D_what) + ".mat")
    mat = scipy.io.loadmat(path_to_data_file)

    for i in range(AN_beginning, AN_end+1):
        rawr = np.array(mat['AN'+str(i)])
        rawr.flatten()
        meow = torch.tensor(rawr)
        print(meow)
        print(len(meow))
        biggie_T[i-AN_beginning-1] = meow
    return biggie_T

print(extract_data('Damaged',3,10,3))


def extract_data_2(path, n, s):
    mat_file = scipy.io.loadmat(path+f'{n}.mat')
    data_np = mat_file[f'AN{s}']
    data_np = data_np.flatten()
    data_tensor = torch.tensor(data_np)
    return data_tensor

def tensor_append(list, x):
    if list == torch.Tensor([0]):
        list = x
    else:
        list = torch.cat((list, x), 0)
    return list



def normalize(a, end=255, intit=True):
    # Normalizing so al values are between 0 and end value
    min = a.min()
    a = a - min
    max = a.max()
    a = a * end / max
    if intit:
        a = a.type(torch.int16)
    return a


def create_matrix(a):
    # Create a normalized matrix of the sample vector
    n = math.floor(a.size(dim=0) ** 0.5)
    m = n
    rest = a.size(dim=0) - n * m
    if (rest != 0):
        a_rest = torch.split(a, (n*m, rest))
        a = a_rest[0]
    a = normalize(a)
    matrix = torch.reshape(a, (n, m))
    return matrix


def generate_samples(sensor_data, n_samples, sample_size, spacing=False):
    # input is one minute of datapoints of one sensor [tensor]
    samples = torch.empty(20)
    if spacing:
        rest = sensor_data.shape[0] - n_samples * sample_size ** 2
        space = rest // n_samples
        b = 0
        e = sample_size**2
        for i in range(n_samples):
            samples[i] = sensor_data[b:e]
            b = e + space
            e += (sample_size**2 + space)
    else:
        for i in range(n_samples):
            samples[i] = sensor_data[(i * sample_size**2):((i + 1) * sample_size**2)]
    for j in range(n_samples):
        samples[i] = create_matrix(samples[i])
    return samples


def visualize(data_matrix):
    # Take the normalized data matrices (that have entries with values ranging from 0 to 255) and visualize these on a grey scale
    plt.imshow(data_matrix, cmap='viridis')
    plt.colorbar()
    plt.show()
    return


if __name__ == '__main__':
    # test your functions here

    # Visualize the data matrix
    test_matrix = np.random.randint(low=0, high=256, size=(224, 224))
    visualize(test_matrix)

    
    print(extract_data('data/Damaged/D', 3, 10, 10))
