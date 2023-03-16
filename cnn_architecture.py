from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten



class CNN(Module):
	def __init__(self, numChannels, classes):
		# call the parent constructor
		super(CNN, self).__init__()
		# initialize first set of CONV => RELU => POOL layers
		self.conv1 = Conv2d(in_channels=numChannels, out_channels=60,
			kernel_size=(5, 5), stride=(2, 2), padding=10)
		self.relu = ReLU()
		self.maxpool1 = MaxPool2d(kernel_size=(2), stride=(2, 2))
		# initialize second set of CONV => RELU => POOL layers
		self.conv2 = Conv2d(in_channels=60, out_channels=50,
			kernel_size=(3, 3), padding=1)
		self.maxpool2 = MaxPool2d(kernel_size=(2), stride=(2, 2))
		self.conv3 = Conv2d(in_channels=50, out_channels=40,
			kernel_size=(3, 3), padding=1)
		self.maxpool3 = MaxPool2d(kernel_size=(2), stride=(2, 2))
		self.conv4 = Conv2d(in_channels=40, out_channels=20,
			kernel_size=(3, 3), padding=1)
		self.maxpool4 = MaxPool2d(kernel_size=(2), stride=(2, 2))

		self.fc1 = Linear(in_features=7*7*20, out_features=400)
		self.fc2 = Linear(in_features=400, out_features=200)
		self.logSoftmax = LogSoftmax(dim=1)

	def forward(self, x):
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
		x = self.relu5(x)
		x = self.fc2(x)
		x = self.relu6(x)
		output = self.logSoftmax(x)
		return output
