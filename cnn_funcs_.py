import math
import torch
import numpy as np
import preprocessing.ppfuncs as pp
from cnn_training import one_iteration as data_train
from cnn_freqtraining import one_iteration as freq_train
import cnn_architecture as arc
from torch import nn



def function(i_max_):
    exp = math.exp(1)
    i_ = torch.arange(0, i_max_)
    return exp**(math.log(2) * i_ / i_max_) - 1


def combining_function(list_0, list_1, f_):
    return list_0 * (1 - f_) + list_1 * f_


def suffle_tensor(list_):
    return list_[torch.randperm(list_.size(dim=0))][:, torch.arange(list_.size(dim=1))][:, :, torch.arange(list_.size(dim=2))][:, :, :, torch.arange(list_.size(dim=3))]


def generate_data(list_data, list_labels):
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
    fun = function(len_com)[:, None, None, None]
    list_new_data = torch.cat((list_new_data, list_1_data[0:len_start]), 0)
    data_com = combining_function(list_1_data[len_start:len_start+len_com], list_0_data[0:len_com], fun)
    list_new_data = torch.cat((list_new_data, data_com), 0)
    list_new_data = torch.cat((list_new_data, list_0_data[len_com:len_com+len_end]), 0)
    list_noise = np.random.normal(0, 1, len(list_new_data))[:, None, None, None]
    list_new_data = list_new_data + list_noise
    return list_new_data


def train_data_model(train_dict, val_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = arc.CreateDataset(train_dict["label"], train_dict["data"])
    val_data = arc.CreateDataset(val_dict["label"], val_dict["data"])

    lr = 0.001
    batch = 50
    epoch = 20
    lf = nn.NLLLoss()
    num_optm = 2

    model, history = data_train(lr, batch, epoch, lf, num_optm, train_data, val_data, device)

    return model

def train_freq_model():
    model, history = freq_train()



if __name__ == "__main__":
    data = torch.load("preprocessing/data_dict.pt")
    print(data["label"][:,1])
    list = generate_data(data["data"], data["label"][:, 1])
    print(list.shape)

