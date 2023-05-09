from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch.utils.data import Dataset
from torch import flatten
import torch
import numpy as np
from functools import reduce


class CreateDataset(Dataset):
	def __init__(self, label_tens, img_tens, transform=None, target_transform=None):
		self.img_labels = label_tens
		self.img_tens = img_tens
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.img_labels)

	def __getitem__(self, idx):
		img_idx = int(self.img_labels[idx][0])
		image = self.img_tens[img_idx]
		label = self.img_labels[idx][1]
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)
		return image, label

	def get_all(self):
		all_images = self.img_tens
		all_labels = self.img_labels
		return all_images, all_labels


class newCNN4(Module):
	def __init__(self, numChannels, classes):
		# call the parent constructor
		super(newCNN4, self).__init__()
		# initialize first set of CONV => RELU => POOL layers
		self.conv1 = Conv2d(in_channels=numChannels, out_channels=60,
			kernel_size=(6, 6), stride=(2, 2), padding=10)
		self.relu = ReLU()
		self.maxpool1 = MaxPool2d(kernel_size=(4), stride=(4, 4))
		# initialize second set of CONV => RELU => POOL layers
		self.conv2 = Conv2d(in_channels=60, out_channels=50,
			kernel_size=(3, 3), padding=1)
		self.maxpool2 = MaxPool2d(kernel_size=(4), stride=(4, 4))
		self.conv3 = Conv2d(in_channels=50, out_channels=20,
			kernel_size=(3, 3), padding=1)
		self.maxpool3 = MaxPool2d(kernel_size=(4), stride=(4, 4))
		#self.conv4 = Conv2d(in_channels=40, out_channels=20,
		#	kernel_size=(3, 3), padding=1)
		#self.maxpool4 = MaxPool2d(kernel_size=(2), stride=(2, 2))

		self.fc1 = Linear(in_features=20, out_features=10)
		self.fc3 = Linear(in_features=10, out_features=classes)
		self.logSoftmax = LogSoftmax(dim=1)

	def forward(self, x):
		# print("Forward running")
		x = self.conv1(x)
		# print(f"test 1{x.shape}")
		x = self.relu(x)
		x = self.maxpool1(x)
		# print(f"test 2{x.shape}")
		x = self.conv2(x)
		# print(f"test 3{x.shape}")
		x = self.relu(x)
		x = self.maxpool2(x)
		#print(f"test 4{x.shape}")
		x = self.conv3(x)
		#print(f"test 5{x.shape}")
		x = self.relu(x)
		x = self.maxpool3(x)
		#print(f"test 6{x.shape}")
		#x = self.conv4(x)
		# print(f"test 7{x.shape}")
		#x = self.relu(x)
		#x = self.maxpool4(x)
		# print(f"test 8{x.shape}")

		x = flatten(x, 1)
		x = self.fc1(x)
		#print(f"test 9{x.shape}")
		# x = self.relu(x)
		#print(f"test 10{x.shape}")
		# x = self.relu(x)
		x = self.fc3(x)
		# x = self.relu(x)
		#print(f"test 11{x.shape}")
		output = self.logSoftmax(x)
		return output

if __name__ == "__main__":
	# 	Create matrix
	num_matrices = 1
	matrix_shape = (224, 224)
	ones_matrices = np.ones((num_matrices,) + matrix_shape)
	print(f"shape of matrix {reduce((lambda x, y: x * y), ones_matrices.shape)}")
	random_matrices = np.random.randint(0,255, (num_matrices,) + matrix_shape)
	# initialize class
	cnn = newCNN4(numChannels=1, classes=2)
	x = torch.from_numpy(ones_matrices).float().unsqueeze(1).repeat(1, 1, 1, 1)
	x2 = torch.from_numpy(random_matrices).float().unsqueeze(1).repeat(1, 1, 1, 1)
	output = cnn.forward(x)
	output2 = cnn.forward(x2)
	print(output)
	print(output2)