from captum.attr import Occlusion
import torch
import cnn_architecture as arc
from cnn_architecture import newCNN4
occlusion = Occlusion(newCNN4(numChannels=1, classes=2))
import numpy as np
from captum.attr import visualization as viz
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sensor_data_damaged = pp.extract_data_2("preprocessing/data/Damaged/D", '2', 9, device=device)
sensor_data_healthy = pp.extract_data_2("preprocessing/data/Healthy/H", '2', 9, device=device)
sensor_samples_damaged = pp.create_samples(sensor_data_damaged, 10, 16700, device=device)
sensor_samples_healthy = pp.create_samples(sensor_data_healthy, 10, 4000, device=device)
spectrogram = pp.spectrogram_2(sensor_samples_healthy[0])
pp.visualize_compare(sensor_samples_healthy, sensor_samples_damaged, 2)
# input_img =
strides = (3, 9, 9)               # smaller = more fine-grained attribution but slower                      # Labrador index in ImageNet
sliding_window_shapes = (3, 45, 45)  # choose size enough to change object appearance
target = 0
baselines = 0
damaged_box = occlusion.attribute(input_img,
                                       strides = strides,
                                       target=target,
                                       sliding_window_shapes=sliding_window_shapes,
                                       baselines=baselines)


target=1,                       # Persian cat index in ImageNet
healthy_box = occlusion.attribute(input_img,
                                       strides = strides,
                                       target=target,
                                       sliding_window_shapes=sliding_window_shapes,
                                       baselines=baselines)

# Convert the compute attribution tensor into an image-like numpy array
damaged_box = np.transpose(damaged_box.squeeze().cpu().detach().numpy(), (1,2,0))

vis_types = ["heat_map", "original_image"]
vis_signs = ["all", "all"] # "positive", "negative", or "all" to show both
# positive attribution indicates that the presence of the area increases the prediction score
# negative attribution indicates distractor areas whose absence increases the score

_ = viz.visualize_image_attr_multiple(damaged_box,
                                      np.array(input_img),
                                      vis_types,
                                      vis_signs,
                                      ["attribution for damaged", "image"],
                                      show_colorbar=True
                                     )


healthy_box = np.transpose(healthy_box.squeeze().cpu().detach().numpy(), (1,2,0))

_ = viz.visualize_image_attr_multiple(healthy_box,
                                      np.array(input_img),
                                      ["heat_map", "original_image"],
                                      ["all", "all"],       # positive/negative attribution or all
                                      ["attribution for healthy", "image"],
                                      show_colorbar=True
                                     )