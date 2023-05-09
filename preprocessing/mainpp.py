# Main file loop for preprocessing

import torch
import ppfuncs as pp


if __name__ == '__main__':
    create_images_ = True
    create_data_set_ = True
    # Set to False if you don't want to print the proces
    print_ = True

    # Define import variables for both Healthy and Damaged
    time_start = 1
    time_end = 10
    sen_start = 3
    sen_end = 10

    # Define image variables for both Healthy and Damaged
    max_images = True

    images_size = 224
    if max_images:
        count_datapoints = 60 * 40 * 1000
        n_images_sensor = int(count_datapoints / (images_size**2))
    else:
        n_images_sensor = 20

    TRAIN_SPLIT = 0.65
    VAL_SPLIT = 0.20
    TEST_SPLIT = 0.15

    if create_images_:
        # Create list of healthy images, by extracting data and creating images with above defined variables
        healthy_image_list = torch.Tensor([])
        path = "data/Healthy/H"
        if print_:
            print("Start on healthy data")
        for i in range(time_start, time_end+1):
            for j in range(sen_start, sen_end+1):
                data = pp.extract_data(path, i, j)
                images = pp.generate_samples(data, n_images_sensor, images_size)
                healthy_image_list = torch.cat((healthy_image_list, images), 0)
            if print_:
                print(f"\tHealthy data at time {i}: Completed")
        if print_:
            print("Healthy data: Completed \n\nStart on Damaged data")

        # Create list of damaged images, by extracting data and creating images with above defined variables
        damaged_image_list = torch.Tensor([])
        path = "data/Damaged/D"
        for i in range(time_start, time_end+1):
            for j in range(sen_start, sen_end+1):
                data = pp.extract_data(path, i, j)
                images = pp.generate_samples(data, n_images_sensor, images_size)
                damaged_image_list = torch.cat((damaged_image_list, images), 0)
            if print_:
                print(f"\tDamaged data at time {i}: Completed")
        if print_:
            print("Damaged data: Completed \n\nStart on saving data")

        # pp.visualize_compare(healthy_image_list.numpy(), damaged_image_list.numpy(), 8)

        data_dict = pp.create_data_dict(healthy_image_list, damaged_image_list)
        torch.save(data_dict, "data_dict.pt")
        if print_:
            print("Saving data: Completed")
    else:
        data_dict = torch.load("data_dict.pt")
        if print_:
            print("Loading in data: Completed")

    if create_data_set_:
        train_data_dict, val_data_dict, test_data_dict = pp.split_data_dict(data_dict, TRAIN_SPLIT, VAL_SPLIT,
                                                                            TEST_SPLIT)
        if print_:
            print("Splitting datasets: Completed")
        torch.save(train_data_dict, "../train_data_dict.pt")
        torch.save(val_data_dict, "../val_data_dict.pt")
        torch.save(test_data_dict, "../test_data_dict.pt")
        if print_:
            print("Saving datasets: Completed")
