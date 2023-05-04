# Main file to preprocess and save the data in separate image files

import torch
import numpy as np
import pandas as pd
import ppfuncs as pp

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Code will be executed on:", device)


# device = torch.device('cuda')

def create_file_name(lab, ti, se, ii):
    while len(str(ti)) < 2:
        ti = "0" + str(ti)
    while len(str(se)) < 2:
        se = "0" + str(se)
    while len(str(ii)) < 3:
        ii = "0" + str(ii)
    file_name = f"image_{lab}_{ti}_{se}_{ii}"
    return file_name


def save_image(list_, img_, label_, time_, sen_, i_):
    file_name = create_file_name(label_, time_, sen_, i_)
    path_ = f'images\{file_name}.pt'
    torch.save(img_, path)
    img_label = np.array([[path_, label_, sen_]])

    list_img_label_ = np.append(list_, img_label, axis=0)
    return list_img_label_


if __name__ == '__main__':
    # Set to False if you don't want to print the proces
    print_ = True

    # Define import variables for both Healthy and Damaged
    time_start = 1
    time_end = 10
    sen_start = 3
    sen_end = 10

    # Define image variables for both Healthy and Damaged
    max_images = True
    n_images_sensor_ = 2
    images_size = 224

    if max_images:
        count_datapoints = 60 * 40 * 1000
        n_images_sensor = int(count_datapoints / (images_size ** 2))
    else:
        n_images_sensor = n_images_sensor_

    path_list = ["data/Healthy/H", "data/Damaged/D"]
    name_list = ["Healthy", "Damaged"]

    img_label_list = np.empty((0, 3))

    for label in [0, 1]:
        name = name_list[label]
        path = path_list[label]

        if print_:
            print(f"Start on {name} data")

        for time in range(time_start, time_end + 1):
            for sen in range(sen_start, sen_end + 1):

                data = pp.extract_data_2(path, time, sen, device=device)
                images_list = pp.generate_samples(data, n_images_sensor, images_size, device=device)
                for i in range(n_images_sensor):
                    img = images_list[i]
                    img_label_list = save_image(img_label_list, img, label, time, sen, i)

            if print_:
                print(f"\t{name} data at time {time}: Completed")
        if print_:
            print(f"{name} data: Completed \n")

    print(img_label_list)
    img_label_pd = pd.DataFrame(img_label_list, columns=["path", "label", "sensors"])
    img_label_pd.to_csv('images/img_labels.cvs', index=False)
