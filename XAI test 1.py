from captum.attr import Occlusion
import torch
import cnn_architecture as arc
from cnn_architecture import newCNN4
import torchvision
import numpy as np
from captum.attr import visualization as viz
from torch.utils.data import DataLoader
from preprocessing import ppfuncs_2 as pp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('TestModel.4fclayers2')
model.eval()
model = model.to(device)
occlusion = Occlusion(model)
path = 'preprocessing/data_dict.pt'

dictionary = torch.load(path)
print(dictionary['data'][dictionary["label"][:,1] == 1][0][0])

def normalize(x):
    mean = torch.mean(x)
    dev = torch.std(x)
    norm = (x-mean)/dev
    return norm

img = dictionary['data'][dictionary["label"][:,1] == 1][0][0]
input_img = normalize(img).float()
input_img = input_img.unsqueeze(0)
input_img = input_img.to(device)

strides = (9,9)           # smaller = more fine-grained attribution but slower                      # Labrador index in ImageNet
sliding_window_shapes = (45,45)

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
                                      np.array(img),
                                      vis_types,
                                      vis_signs,
                                      ["attribution for damaged", "image"],
                                      show_colorbar=True
                                     )


healthy_box = np.transpose(healthy_box.squeeze().cpu().detach().numpy(), (1,2,0))

_ = viz.visualize_image_attr_multiple(healthy_box,
                                      np.array(img),
                                      ["heat_map", "original_image"],
                                      ["all", "all"],       # positive/negative attribution or all
                                      ["attribution for healthy", "image"],
                                      show_colorbar=True
                                     )