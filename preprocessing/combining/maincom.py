import math
import torch
import preprocessing.ppfuncs as pp


def function(i_, i_max_):
    return math.exp(math.log(2) * i_ / i_max_) - 1


def combining_function(list_0, list_1, i_count, i_round, i_max):
    i_start = i_count * i_round
    data_list = torch.Tensor([])
    for i in range(i_start, i_start+len(list_0)):
        f_ = function(i, i_max)
        data_ = (1 - f_) * list_0[i] + f_ * list_1[i]
        data_list = torch.cat((data_list, torch.Tensor([data_])), 0)
    return data_list


if __name__ == '__main__':
    print_ = True

    sen_start = 3
    sen_end = 10

    max_images = True
    images_size = 224
    if max_images:
        count_datapoints = 60 * 40 * 1000
        n_images_sensor = int(count_datapoints / (images_size**2))
    else:
        n_images_sensor = 20

    path = "../data/"
    path_list = [f"{path}Healthy/H", f"{path}Damaged/D"]

    image_list = torch.Tensor([])

    ind_count = -1
    ind_round = 40 * 1000 * 60
    ind_max = ind_round * 3

    for sen in range(sen_start, sen_end+1):
        for time in range(1, 3+1):
            data = pp.extract_data(path_list[0], time, sen)
            images = pp.generate_samples(data, n_images_sensor, images_size)
            image_list = torch.cat((image_list, images), 0)
        for time in range(4, 6+1):
            ind_count += 1
            data_0 = pp.extract_data(path_list[0], time, sen)
            data_1 = pp.extract_data(path_list[1], time, sen)
            data = combining_function(data_0, data_1, ind_count, ind_round, ind_max)
            images = pp.generate_samples(data, n_images_sensor, images_size)
            image_list = torch.cat((image_list, images), 0)
        for time in range(7, 9+1):
            data = pp.extract_data(path_list[1], time, sen)
            images = pp.generate_samples(data, n_images_sensor, images_size)
            image_list = torch.cat((image_list, images), 0)
        torch.save(image_list, f"combined_sensor{sen}.pt")
    if print_:
        print(f"sensor {sen}: Completed")
