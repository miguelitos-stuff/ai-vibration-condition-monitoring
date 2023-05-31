from captum.attr import Occlusion
import torch
import cnn_architecture as arc
from cnn_architecture import newCNN4
occlusion = Occlusion(newCNN4(numChannels=1, classes=2))
import numpy as np
from captum.attr import visualization as viz
from torchvision import transforms
from torch.utils.data import DataLoader
from preprocessing import ppfuncs_2 as pp

# train_data = torch.load('train_data_dict.pt')
# train_data = arc.CreateDataset(train_data["label"], train_data["data"])
# print("Size of test dataset:", len(train_data))
#
# val_data = torch.load('val_data_dict.pt')
# val_data = arc.CreateDataset(val_data["label"], val_data["data"])
# print("Size of validation dataset:", len(val_data))
#
# test_data = torch.load('test_data_dict.pt')
# test_data = arc.CreateDataset(test_data["label"], test_data["data"])
# print("Size of testing dataset:", len(test_data))
#
#
# BATCH_SIZE = 50
# trainDataLoader = DataLoader(train_data, shuffle=True,batch_size=BATCH_SIZE)
# valDataLoader = DataLoader(val_data, batch_size=BATCH_SIZE)
# testDataLoader = DataLoader(test_data, batch_size=BATCH_SIZE)

dictionary = torch.load('preprocessing/data_dict.pt')
#print(dictionary['data'][dictionary["label"][:,1] == 1][0][0])


# center_crop = transforms.Compose([
#  transforms.Resize(256),
#  transforms.CenterCrop(224),
# ])

def normalize(x):
    mean = torch.mean(x)
    dev = torch.std(x)
    norm = (x-mean)/dev
    return norm

input_img = normalize(dictionary['data'][dictionary["label"][:,1] == 1][:][0])
input_img = input_img.to(torch.float32).unsqueeze(0)
#print(input_img)
#print(input_img.size())
img = dictionary['data'][dictionary["label"][:,1] == 1][0][0]
strider = tuple([112])             # smaller = more fine-grained attribution but slower                      # Labrador index in ImageNet
sliding_window_shapes = tuple([1,224,224])  # choose size enough to change object appearance
target = 0
baselines = 0
damaged_box = occlusion.attribute(input_img,
                                       strides = strider,
                                       target=target,
                                       sliding_window_shapes=sliding_window_shapes,
                                       baselines=baselines)

print(damaged_box)
print(damaged_box.shape)
target=1,                       # Persian cat index in ImageNet
healthy_box = occlusion.attribute(input_img,
                                       strides = strider,
                                       target=target,
                                       sliding_window_shapes=sliding_window_shapes,
                                       baselines=baselines)

# Convert the compute attribution tensor into an image-like numpy array
#print(damaged_box.size())
#print(damaged_box)
print('hi')
print(damaged_box.shape)
print(img.shape)
#damaged_box = np.transpose(damaged_box.squeeze().cpu().detach().numpy(), (1, 2, 0))
#img = np.transpose(img.squeeze().cpu().detach().numpy(), (1, 2, 0))
print(damaged_box.shape)
print(img.shape)


vis_types = ["heat_map", "original_image"]
vis_signs = ["all", "all"] # "positive", "negative", or "all" to show both
# positive attribution indicates that the presence of the area increases the prediction score
# negative attribution indicates distractor areas whose absence increases the score

_ = viz.visualize_image_attr_multiple(damaged_box,
                                      np.array(img),
                                      vis_types,
                                      vis_signs,
                                      ["attribution for damaged", "image"],
                                      show_colorbar=True
                                     )


healthy_box = np.transpose(healthy_box.squeeze().cpu().detach().numpy(), (0,2,3,1))

_ = viz.visualize_image_attr_multiple(healthy_box,
                                      np.array(img),
                                      ["heat_map", "original_image"],
                                      ["all", "all"],       # positive/negative attribution or all
                                      ["attribution for healthy", "image"],
                                      show_colorbar=True
                                     )