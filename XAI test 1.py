from captum.attr import Occlusion
from cnn_architecture import newCNN4
occlusion = Occlusion(newCNN4(numChannels=1, classes=2))
import numpy as np
from captum.attr import visualization as viz

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