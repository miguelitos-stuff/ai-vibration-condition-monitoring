from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch.utils.data import Dataset
from torch import flatten
from torch.utils.data import DataLoader
import time
import torch
import numpy as np


class CreateDataset(Dataset):
	def __init__(self, label_tens, img_tens):
		self.img_labels = label_tens
		self.img_tens = img_tens.float()

	def __len__(self):
		return len(self.img_labels)

	def __getitem__(self, idx):
		img_idx = int(self.img_labels[idx][0])
		image = self.img_tens[img_idx]
		label = self.img_labels[idx][1]
		return image, label


class CNN(Module):
	def __init__(self, num_channels=1, classes=2):
		# call the parent constructor
		super(CNN, self).__init__()
		# initialize sets of CONV => RELU => POOL layers
		self.conv1 = Conv2d(in_channels=num_channels, out_channels=60, kernel_size=(5, 5), stride=(2, 2), padding=10)
		self.relu = ReLU()
		self.maxpool1 = MaxPool2d(kernel_size=2, stride=(2, 2))
		self.conv2 = Conv2d(in_channels=60, out_channels=50, kernel_size=(3, 3), padding=1)
		self.maxpool2 = MaxPool2d(kernel_size=2, stride=(2, 2))
		self.conv3 = Conv2d(in_channels=50, out_channels=40, kernel_size=(3, 3), padding=1)
		self.maxpool3 = MaxPool2d(kernel_size=2, stride=(2, 2))
		self.conv4 = Conv2d(in_channels=40, out_channels=20, kernel_size=(3, 3), padding=1)
		self.maxpool4 = MaxPool2d(kernel_size=2, stride=(2, 2))
		# initialize MLP
		self.fc1 = Linear(in_features=980, out_features=400)
		self.fc2 = Linear(in_features=400, out_features=200)
		self.fc3 = Linear(in_features=200, out_features=classes)
		self.logSoftmax = LogSoftmax(dim=1)

	def forward(self, x):
		# print("Forward running")
		x = self.conv1(x)
		x = self.relu(x)
		x = self.maxpool1(x)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.maxpool2(x)
		x = self.conv3(x)
		x = self.relu(x)
		x = self.maxpool3(x)
		x = self.conv4(x)
		x = self.relu(x)
		x = self.maxpool4(x)

		x = flatten(x, 1)
		x = self.fc1(x)
		# x = self.relu(x)
		x = self.fc2(x)
		# x = self.relu(x)
		x = self.fc3(x)
		# x = self.relu(x)
		output = self.logSoftmax(x)
		return output


class NewCNN(Module):
	def __init__(self, num_channels, classes):
		# call the parent constructor
		super(NewCNN, self).__init__()
		# initialize sets of CONV => RELU => POOL layers
		self.conv1 = Conv2d(in_channels=num_channels, out_channels=60, kernel_size=(6, 6), stride=(2, 2), padding=10)
		self.relu = ReLU()
		self.maxpool1 = MaxPool2d(kernel_size=2, stride=(2, 2))
		self.conv2 = Conv2d(in_channels=60, out_channels=50, kernel_size=(3, 3), padding=1)
		self.maxpool2 = MaxPool2d(kernel_size=2, stride=(2, 2))
		self.conv3 = Conv2d(in_channels=50, out_channels=20, kernel_size=(3, 3), padding=1)
		self.maxpool3 = MaxPool2d(kernel_size=4, stride=(4, 4))
		# self.conv4 = Conv2d(in_channels=40, out_channels=20, kernel_size=(3, 3), padding=1)
		# self.maxpool4 = MaxPool2d(kernel_size=(2), stride=(2, 2))
		# initialize MLP
		self.fc1 = Linear(in_features=980, out_features=400)
		self.fc2 = Linear(in_features=400, out_features=200)
		self.fc3 = Linear(in_features=200, out_features=classes)
		self.logSoftmax = LogSoftmax(dim=1)

	def forward(self, x):
		# print("Forward running")
		x = self.conv1(x)
		x = self.relu(x)
		x = self.maxpool1(x)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.maxpool2(x)
		x = self.conv3(x)
		x = self.relu(x)
		x = self.maxpool3(x)
		# x = self.conv4(x)
		# x = self.relu(x)
		# x = self.maxpool4(x)

		x = flatten(x, 1)
		x = self.fc1(x)
		# x = self.relu(x)
		x = self.fc2(x)
		# x = self.relu(x)
		x = self.fc3(x)
		# x = self.relu(x)
		output = self.logSoftmax(x)
		return output


class NewCNN2(Module):
	def __init__(self, num_channels, classes):
		# call the parent constructor
		super(NewCNN2, self).__init__()
		# initialize sets of CONV => RELU => POOL layers
		self.conv1 = Conv2d(in_channels=num_channels, out_channels=60, kernel_size=(6, 6), stride=(2, 2), padding=10)
		self.relu = ReLU()
		self.maxpool1 = MaxPool2d(kernel_size=2, stride=(2, 2))
		self.conv2 = Conv2d(in_channels=60, out_channels=50, kernel_size=(3, 3), padding=1)
		self.maxpool2 = MaxPool2d(kernel_size=4, stride=(4, 4))
		self.conv3 = Conv2d(in_channels=50, out_channels=20, kernel_size=(3, 3), padding=1)
		self.maxpool3 = MaxPool2d(kernel_size=4, stride=(4, 4))
		# self.conv4 = Conv2d(in_channels=40, out_channels=20, kernel_size=(3, 3), padding=1)
		# self.maxpool4 = MaxPool2d(kernel_size=(2), stride=(2, 2))
		# initialize MLP
		self.fc1 = Linear(in_features=180, out_features=100)
		self.fc2 = Linear(in_features=100, out_features=50)
		self.fc3 = Linear(in_features=50, out_features=classes)
		self.logSoftmax = LogSoftmax(dim=1)

	def forward(self, x):
		# print("Forward running")
		x = self.conv1(x)
		x = self.relu(x)
		x = self.maxpool1(x)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.maxpool2(x)
		x = self.conv3(x)
		x = self.relu(x)
		x = self.maxpool3(x)
		# x = self.conv4(x)
		# x = self.relu(x)
		# x = self.maxpool4(x)

		x = flatten(x, 1)
		x = self.fc1(x)
		# x = self.relu(x)
		x = self.fc2(x)
		# x = self.relu(x)
		x = self.fc3(x)
		# x = self.relu(x)
		output = self.logSoftmax(x)
		return output


class NewCNN3(Module):
	def __init__(self, num_channels, classes):
		# call the parent constructor
		super(NewCNN3, self).__init__()
		# initialize sets of CONV => RELU => POOL layers
		self.conv1 = Conv2d(in_channels=num_channels, out_channels=60, kernel_size=(5, 5), stride=(2, 2), padding=10)
		self.relu = ReLU()
		self.maxpool1 = MaxPool2d(kernel_size=4, stride=(4, 4))
		self.conv2 = Conv2d(in_channels=60, out_channels=50, kernel_size=(3, 3), padding=1)
		self.maxpool2 = MaxPool2d(kernel_size=4, stride=(4, 4))
		self.conv3 = Conv2d(in_channels=50, out_channels=20, kernel_size=(3, 3), padding=1)
		self.maxpool3 = MaxPool2d(kernel_size=4, stride=(4, 4))
		# self.conv4 = Conv2d(in_channels=40, out_channels=20,
		# kernel_size=(3, 3), padding=1)
		# self.maxpool4 = MaxPool2d(kernel_size=(2), stride=(2, 2))
		# initialize MLP
		self.fc1 = Linear(in_features=20, out_features=10)
		self.fc2 = Linear(in_features=10, out_features=4)
		self.fc3 = Linear(in_features=4, out_features=classes)
		self.logSoftmax = LogSoftmax(dim=1)

	def forward(self, x):
		# print("Forward running")
		x = self.conv1(x)
		x = self.relu(x)
		x = self.maxpool1(x)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.maxpool2(x)
		x = self.conv3(x)
		x = self.relu(x)
		x = self.maxpool3(x)
		# x = self.conv4(x)
		# x = self.relu(x)
		# x = self.maxpool4(x)

		x = flatten(x, 1)
		x = self.fc1(x)
		# x = self.relu(x)
		x = self.fc2(x)
		# x = self.relu(x)
		x = self.fc3(x)
		# x = self.relu(x)
		output = self.logSoftmax(x)
		return output


class NewCNN4(Module):
	def __init__(self, num_channels=1, classes=2):
		# call the parent constructor
		super(NewCNN4, self).__init__()
		# initialize sets of CONV => RELU => POOL layers
		self.conv1 = Conv2d(in_channels=num_channels, out_channels=60, kernel_size=(6, 6), stride=(2, 2), padding=10)
		self.relu = ReLU()
		self.maxpool1 = MaxPool2d(kernel_size=4, stride=(4, 4))
		self.conv2 = Conv2d(in_channels=60, out_channels=50, kernel_size=(3, 3), padding=1)
		self.maxpool2 = MaxPool2d(kernel_size=4, stride=(4, 4))
		self.conv3 = Conv2d(in_channels=50, out_channels=20, kernel_size=(3, 3), padding=1)
		self.maxpool3 = MaxPool2d(kernel_size=4, stride=(4, 4))
		# self.conv4 = Conv2d(in_channels=40, out_channels=20, kernel_size=(3, 3), padding=1)
		# self.maxpool4 = MaxPool2d(kernel_size=2, stride=(2, 2))
		# initialize MLP
		self.fc1 = Linear(in_features=20, out_features=10)
		self.fc3 = Linear(in_features=10, out_features=classes)
		self.logSoftmax = LogSoftmax(dim=1)

	def forward(self, x):
		# print("Forward running")
		x = self.conv1(x)
		x = self.relu(x)
		x = self.maxpool1(x)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.maxpool2(x)
		x = self.conv3(x)
		x = self.relu(x)
		x = self.maxpool3(x)
		# x = self.conv4(x)
		# x = self.relu(x)
		# x = self.maxpool4(x)

		x = flatten(x, 1)
		x = self.fc1(x)
		# x = self.relu(x)
		# x = self.relu(x)
		x = self.fc3(x)
		# x = self.relu(x)
		output = self.logSoftmax(x)
		return output


class FreqCNN(Module):
	def __init__(self, num_channels, classes):
		# call the parent constructor
		super(FreqCNN, self).__init__()
		# initialize sets of CONV => RELU => POOL layers
		self.conv1 = Conv2d(in_channels=num_channels, out_channels=30, kernel_size=(7, 3), stride=(6, 2), padding=10)
		self.relu = ReLU()
		self.maxpool1 = MaxPool2d(kernel_size=4, stride=(4, 4))
		self.conv2 = Conv2d(in_channels=30, out_channels=10, kernel_size=(3, 3), padding=1)
		self.maxpool2 = MaxPool2d(kernel_size=2, stride=(2, 2))
		# self.conv3 = Conv2d(in_channels=50, out_channels=20, kernel_size=(3, 3), padding=1)
		# self.maxpool3 = MaxPool2d(kernel_size=4, stride=(4, 4))
		# self.conv4 = Conv2d(in_channels=40, out_channels=20, kernel_size=(3, 3), padding=1)
		# self.maxpool4 = MaxPool2d(kernel_size=2, stride=(2, 2))
		# initialize MLP
		self.fc1 = Linear(in_features=90, out_features=4)
		# self.fc2 = Linear(in_features=40, out_features=4)
		self.fc3 = Linear(in_features=4, out_features=classes)
		self.logSoftmax = LogSoftmax(dim=1)

	def forward(self, x):
		# print("Forward running")
		x = self.conv1(x)
		x = self.relu(x)
		x = self.maxpool1(x)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.maxpool2(x)
		# x = self.conv3(x)
		# x = self.relu(x)
		# x = self.maxpool3(x)
		# x = self.conv4(x)
		# x = self.relu(x)
		# x = self.maxpool4(x)

		x = flatten(x, 1)
		x = self.fc1(x)
		# x = self.relu(x)
		# x = self.fc2(x)
		# x = self.relu(x)
		x = self.fc3(x)
		# x = self.relu(x)
		output = self.logSoftmax(x)
		return output


if __name__ == "__main__":
	# 	Create matrix
	num_matrices = 1
	matrix_shape = (129, 30)
	ones_matrices = np.ones((num_matrices,) + matrix_shape)
	print(ones_matrices.shape)
	random_matrices = np.random.randint(0, 255, (num_matrices,) + matrix_shape)
	# initialize class
	cnn = FreqCNN(1, 2)
	x1 = torch.from_numpy(ones_matrices).float().unsqueeze(1).repeat(1, 1, 1, 1)
	x2 = torch.from_numpy(random_matrices).float().unsqueeze(1).repeat(1, 1, 1, 1)
	model_output = cnn.forward(x1)
	# output2 = cnn.forward(x2)
	print(model_output)
	# print(output2)

	# # 	Create matrix
	# num_matrices = 1
	# matrix_shape = (224, 224)
	# ones_matrices = np.ones((num_matrices,) + matrix_shape)
	# print(ones_matrices.shape)
	# random_matrices = np.random.randint(0,255, (num_matrices,) + matrix_shape)
	# # initialize class
	# cnn = CNN(num_channels=1, classes=2)
	# x1 = torch.from_numpy(ones_matrices).float().unsqueeze(1).repeat(1, 1, 1, 1)
	# x2 = torch.from_numpy(random_matrices).float().unsqueeze(1).repeat(1, 1, 1, 1)
	# output = cnn.forward(x1)
	# output2 = cnn.forward(x2)
	# print(output)
	# print(output2)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	test_data = torch.load('test_data_dict.pt')
	test_data = CreateDataset(test_data["label"], test_data["data"])

	testDataLoader = DataLoader(test_data, shuffle=True, batch_size=50)
	model = CNN(num_channels=1, classes=2).to(device)
	computation_time = 0
	print(len(testDataLoader))
	lengths = 0

	for (x1, y1) in testDataLoader:
		# send the input to the device
		y1 = y1.type(torch.LongTensor)
		(x1, y1) = (x1.to(device), y1.to(device))
		# make the predictions and calculate the validation loss
		start_compute = time.perf_counter()
		pred = model(x1)
		end_compute = time.perf_counter()
		computation_time += (end_compute - start_compute)
		lengths += len(x1)
	avg_compute_time = computation_time/lengths

	print(avg_compute_time)

# 1 0.00095142194139895
# 2 0.0012046955453115515
