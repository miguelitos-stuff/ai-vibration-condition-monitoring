import ppfuncs as pp
import torch
import numpy as np

# Load the file
data_dict = torch.load("data_dict.pt")

images = data_dict["data"]
labels = data_dict["label"]

# pp.visualize(images[6])

N_compare = 3
n = 0
healthy_image_list = torch.Tensor([])
# find N random healthy labels and put in list
while n < N_compare:
    rint = np.random.randint(0, len(labels))
    if labels[rint] == 1:
        healthy_image_list = torch.cat((healthy_image_list, torch.unsqueeze(images[rint], 0)) ,0)
        n += 1
n = 0
damaged_image_list = torch.Tensor([])
# find N random healthy labels and put in list
while n < N_compare:
    rint = np.random.randint(0, len(labels))
    if labels[rint] == 0:
        damaged_image_list = torch.cat((damaged_image_list, torch.unsqueeze(images[rint], 0)) ,0)
        n += 1
pp.visualize_compare(healthy_image_list, damaged_image_list, N_compare)


