import math
import torch
import numpy as np
from cnn_training import one_iteration as ampl_train
from cnn_freqtraining import one_iteration as freq_train
import cnn_architecture as arc
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch.utils.data import DataLoader


def function(i_max_):
    exp = math.exp(1)
    i_ = torch.arange(0, i_max_)
    return exp**(math.log(2) * i_ / i_max_) - 1


def combining_function(list_0, list_1, f_):
    return list_0 * (1 - f_) + list_1 * f_


def suffle_tensor(list_):
    return list_[torch.randperm(list_.size(dim=0))][:, torch.arange(list_.size(dim=1))][:, :, torch.arange(list_.size(dim=2))][:, :, :, torch.arange(list_.size(dim=3))]


def generate_data(list_data, list_labels, noise=0):
    list_1_data = list_data[list_labels == 1]
    list_0_data = list_data[list_labels == 0]
    list_1_data = suffle_tensor(list_1_data)
    list_0_data = suffle_tensor(list_0_data)
    len_1 = len(list_1_data)
    len_0 = len(list_0_data)
    len_com = int(min(len_1, len_0) * 2 / 3)
    len_start = len_1 - len_com
    len_end = len_0 - len_com
    list_new_data = torch.Tensor([])
    list_f = torch.Tensor([])
    function_ = function(len_com)
    plt.figure()
    plt.plot(np.arange(len_com), function_)
    plt.savefig("test.png")
    fun = function_[:, None, None, None]
    list_new_data = torch.cat((list_new_data, list_1_data[0:len_start]), 0)
    list_f = torch.cat((list_f, torch.ones(len_start)), 0)
    data_com = combining_function(list_1_data[len_start:len_start+len_com], list_0_data[0:len_com], fun)
    list_new_data = torch.cat((list_new_data, data_com), 0)
    list_f = torch.cat((list_f, 1 - function_), 0)
    list_new_data = torch.cat((list_new_data, list_0_data[len_com:len_com+len_end]), 0)
    list_f = torch.cat((list_f, torch.zeros(len_end)), 0)
    if noise != 0:
        std = torch.std(list_new_data)
        print(std)
        list_noise = np.random.normal(0, std/noise, size=tuple(list_new_data.size()))
        print(list_noise[0])
        list_new_data = list_new_data + list_noise
    return list_new_data, list_f


def train_ampl_model(train_dict, val_dict, test_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = arc.CreateDataset(train_dict["label"], train_dict["data"])
    val_data = arc.CreateDataset(val_dict["label"], val_dict["data"])
    test_data = arc.CreateDataset(test_dict["label"], test_dict["data"])

    lr = 0.001
    batch = 50
    epoch = 20
    lf = nn.NLLLoss()
    optm = 1

    model, history = ampl_train(lr, batch, epoch, lf, optm, train_data, val_data, test_data, device)
    return model


def train_freq_model(train_dict, val_dict, test_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = arc.CreateDataset(train_dict["label"], train_dict["data"])
    val_data = arc.CreateDataset(val_dict["label"], val_dict["data"])
    test_data = arc.CreateDataset(test_dict["label"], test_dict["data"])

    lr = 0.001
    batch = 50
    epoch = 20
    lf = nn.NLLLoss()
    optm = 1

    model, history = freq_train(lr, batch, epoch, lf, optm, train_data, val_data, test_data, device)
    return model


def run_save_graph(model, data_, name, f_, title):
    result = model.forward(data_)
    # print(result)
    result_ = torch.split(result, 1, 1)
    result_0 = 10**result_[0].detach().numpy()
    result_1 = 10**result_[1].detach().numpy()
    comb = (np.add(result_1, -1 * result_0) + 1)/2
    plt.figure(figsize=(5, 10))
    plt.xlim((0, len(comb)))
    plt.yticks([0.0, 0.25, 0.50, 0.75, 1.0], ["0.00", "0.25", "0.50", "0.75", "1.00"])
    plt.grid(color='0.8', linestyle='-', linewidth=0.5)
    plt.plot(np.arange(len(comb)), comb)
    plt.plot(np.arange(len(comb)), f_)
    plt.title(title)
    plt.ylabel("Probability [-]")
    plt.xlabel("Images [-]")
    plt.savefig("graphs\\"+name+".png")
    # comb = (np.add(result_1, -1 * result_0) + 1) / 2
    # # result = 10 ** result.abs().detach().numpy()
    # # output = np.argmax(result, axis=1)
    # plt.figure(figsize=(5, 10))
    # plt.xlim((0, len(comb)))
    # plt.xticks([0, len(comb) * 3 / 10, len(comb) * 5 / 10, len(comb) * 7 / 10, len(comb)], [0, 3, 5, 7, 10])
    # # plt.yticks([0.0, 0.25, 0.50, 0.75, 1.0], ["0.00", "0.25", "0.50", "0.75", "1.00"])
    # plt.grid(color='0.8', linestyle='-', linewidth=0.5)
    # plt.plot(np.arange(len(comb)), comb)
    # plt.ylabel("Probability [-]")
    # plt.xlabel("Time [min]")
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # ax1.plot(np.arange(len(result_0)), result_0)
    # ax1.set_title("Prop Damaged")
    # ax2.plot(np.arange(len(result_1)), result_1)
    # ax2.set_title("Prop Healthy")
    # ax3.plot(np.arange(len(comb)), comb)
    # ax3.set_title("Comb prop Healthy")
    # plt.savefig("graphs\graphs_" + name + ".png")
    return comb

def set_to_dict(data_, num_, ):
    dataloader = DataLoader(data_, batch_size=num_, shuffle=True)
    features, labels = next(iter(dataloader))
    ind = torch.arange(0, len(features))
    labels = torch.stack((ind, labels), -1)
    data_dict_ = {"data": features, "label": labels}
    return data_dict_


def split_train_val_dict(data_dict_, split_):
    all_data = arc.CreateDataset(data_dict_["label"], data_dict_["data"])
    num_train = int(round(len(all_data) * (1-split_), 0))
    num_val = int(round(len(all_data) * split_, 0))
    (train_data_, val_data_) = random_split(all_data, [num_train, num_val],
                                                        generator=torch.Generator().manual_seed(42))
    train_data_dict_ = set_to_dict(train_data_, num_train)
    val_data_dict_ = set_to_dict(val_data_, num_val)
    return train_data_dict_, val_data_dict_


if __name__ == "__main__":
    print("hello")
