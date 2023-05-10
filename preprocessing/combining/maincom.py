import math
import torch
import preprocessing.ppfuncs as pp


def function(i_max_):
    exp = math.exp(1)
    i_ = torch.arange(0, i_max_)
    return exp**(math.log(2) * i_ / i_max_) - 1


def combining_function(list_0, list_1, f_):
    return torch.mul((1 - f_), list_0) + torch.mul(f_, list_1)


if __name__ == '__main__':
    print_ = True

    sen_start = 3
    sen_end = 3

    time_0 = 1
    time_1 = 3
    time_2 = 7
    time_3 = 10

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
    ind_max = ind_round * (time_2-time_1)

    fun = torch.split(function(ind_max), ind_round)

    for sen in range(sen_start, sen_end+1):
        for time in range(time_0, time_1+1):
            data = pp.extract_data(path_list[0], time, sen)
            images = pp.generate_samples(data, n_images_sensor, images_size)
            image_list = torch.cat((image_list, images), 0)
        for time in range(time_1+1, time_2+1):
            ind_count += 1
            data_0 = pp.extract_data(path_list[0], time, sen)
            data_1 = pp.extract_data(path_list[1], time, sen)
            data = combining_function(data_0, data_1, fun[ind_count])
            images = pp.generate_samples(data, n_images_sensor, images_size)
            image_list = torch.cat((image_list, images), 0)
        for time in range(time_2+1, time_3+1):
            data = pp.extract_data(path_list[1], time, sen)
            images = pp.generate_samples(data, n_images_sensor, images_size)
            image_list = torch.cat((image_list, images), 0)
        torch.save(image_list, f"combined_sensor{sen}.pt")
        if print_:
            print(f"sensor {sen}: Completed")
