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
        path_h = "data/Healthy/H"
        path_d = "data/Damaged/D"

        if print_:
            print("Start on data")
        for j in range(sen_start, sen_end+1):
            healthy_image_list = torch.Tensor([])
            damaged_image_list = torch.Tensor([])
            for i in range(time_start, time_end+1):
                data = pp.extract_data(path_h, i, j)
                images = pp.generate_samples(data, n_images_sensor, images_size)
                healthy_image_list = torch.cat((healthy_image_list, images), 0)
                data = pp.extract_data(path_d, i, j)
                images = pp.generate_samples(data, n_images_sensor, images_size)
                damaged_image_list = torch.cat((damaged_image_list, images), 0)
            if print_:
                print(f"\tHealthy data at sensor {j}: Completed")
            data_dict = pp.create_data_dict(healthy_image_list, damaged_image_list)
            torch.save(data_dict, f"../data_dict/AMED_sensor_{j}.pt")
        if print_:
            print("Data: Completed")

