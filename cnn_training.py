# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from outdated_scripts.cnn_architecture2 import LeNet
from cnn_architecture import CNN
import cnn_architecture as arc
#from preprocessing import 'data_dict.pt'
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch.optim import SGD
from torch.optim import LBFGS
from torch.optim import Adamax
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
import json
import torch
import time
import os


def one_iteration(INIT_LR, BATCH_SIZE, EPOCHS, lossFn, optm, trainData, testData, device):
	# define the train and val splits
	TRAIN_SPLIT = 0.75
	VAL_SPLIT = 1 - TRAIN_SPLIT
	# set the device we will be using to train the model
	print("Pytorch CUDA Version is available:", torch.cuda.is_available())

	# load the KMNIST dataset
	print("[INFO] loading the dataset...")
	#trainData = KMNIST(root="data", train=True, download=True,
		#transform=ToTensor())
	#testData = KMNIST(root="data", train=False, download=True,
		#transform=ToTensor())

	# Change this to load the tensors

	# calculate the train/validation split
	print("[INFO] generating the train/validation split...")
	numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
	numValSamples = int(len(trainData) * VAL_SPLIT)
	(trainData, valData) = random_split(trainData,
		[numTrainSamples, numValSamples],
		generator=torch.Generator().manual_seed(42))

	# initialize the train, validation, and test data loaders
	trainDataLoader = DataLoader(trainData, shuffle=True,
		batch_size=BATCH_SIZE)
	valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE)
	testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)

	# calculate steps per epoch for training and validation set
	trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
	valSteps = len(valDataLoader.dataset) // BATCH_SIZE

	# initialize the LeNet model
	print("[INFO] initializing the LeNet model...")
	model = CNN(
		numChannels=1,
		classes=2).to(device)
	# initialize a dictionary to store training history
	H = {
		"train_loss": [],
		"train_acc": [],
		"val_loss": [],
		"val_acc": []
	}
	if optm == 0:
		opt = Adam(model.parameters(), lr=learning_rate)
	elif optm == 1:
		opt = SGD(model.parameters(), lr=learning_rate)
	elif optm == 2:
		opt = LBFGS(model.parameters(),lr=learning_rate)
	elif optm == 3:
		opt = Adamax(model.parameters(), lr=learning_rate)
	# measure how long training is going to take
	print("[INFO] training the network...")
	startTime = time.time()

	# loop over our epochs
	for e in range(0, EPOCHS):
		# set the model in training mode
		model.train()
		# initialize the total training and validation loss
		totalTrainLoss = 0
		totalValLoss = 0
		# initialize the number of correct predictions in the training
		# and validation step
		trainCorrect = 0
		valCorrect = 0
		# loop over the training set
		for (x, y) in trainDataLoader:
			# send the input to the device
			(x, y) = (x.to(device), y.to(device))
			# perform a forward pass and calculate the training loss
			pred = model(x)
			loss = lossFn(pred, y)
			# zero out the gradients, perform the backpropagation step,
			# and update the weights
			opt.zero_grad()
			loss.backward()
			opt.step()
			# add the loss to the total training loss so far and
			# calculate the number of correct predictions
			totalTrainLoss += loss
			trainCorrect += (pred.argmax(1) == y).type(
				torch.float).sum().item()

		# switch off autograd for evaluation
		with torch.no_grad():
			# set the model in evaluation mode
			model.eval()
			# loop over the validation set
			for (x, y) in valDataLoader:
				# send the input to the device
				(x, y) = (x.to(device), y.to(device))
				# make the predictions and calculate the validation loss
				pred = model(x)
				totalValLoss += lossFn(pred, y)
				# calculate the number of correct predictions
				valCorrect += (pred.argmax(1) == y).type(
					torch.float).sum().item()

		# calculate the average training and validation loss
		avgTrainLoss = totalTrainLoss / trainSteps
		avgValLoss = totalValLoss / valSteps
		# calculate the training and validation accuracy
		trainCorrect = trainCorrect / len(trainDataLoader.dataset)
		valCorrect = valCorrect / len(valDataLoader.dataset)
		# update our training history
		H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
		H["train_acc"].append(trainCorrect)
		H["val_loss"].append(avgValLoss.cpu().detach().numpy())
		H["val_acc"].append(valCorrect)
		# print the model training and validation information
		print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
		print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
			avgTrainLoss, trainCorrect))
		print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
			avgValLoss, valCorrect))

	# finish measuring how long training took
	endTime = time.time()
	print("[INFO] total time taken to train the model: {:.2f}s".format(
		endTime - startTime))
	# we can now evaluate the network on the test set
	print("[INFO] evaluating network...")
	# turn off autograd for testing evaluation
	testCorrect = 0
	totalTestLoss = 0
	with torch.no_grad():
		# set the model in evaluation mode
		model.eval()

		# initialize a list to store our predictions
		preds = []
		# loop over the test set
		for (x, y) in testDataLoader:
			# send the input to the device
			x = x.to(device)
			# make the predictions and add them to the list
			pred = model(x)
			preds.extend(pred.argmax(axis=1).cpu().numpy())
			totalTestLoss += lossFn(pred, y)
			testCorrect += (pred.argmax(1) == y).type(
				torch.float).sum().item()
		test_acc = testCorrect / len(testDataLoader.dataset)
		testSteps = len(testDataLoader.dataset)
		avgTestLoss = totalTestLoss / testSteps
	# generate a classification report
	test_results = classification_report(testData.targets.cpu().numpy(),np.array(preds), target_names=testData.classes)
	H["test_results"] = test_results
	return model, (endTime - startTime), accuracy, H


def ranking_system():
	# assign directory
	directory = 'CNNModels'

	for filename in os.listdir(directory):
		f = os.path.join(directory, filename)
		# checking if it is a file
		if os.path.isfile(f):
			print(f)

ranking_system()

def graph_model_losses(filenames, figure_name):
	plt.clf()
	plt.style.use("ggplot")
	plt.figure()
	for filename in filenames:
		history = open(filename)
		plt.plot(history["train_loss"], label="train_loss")
		plt.plot(history["val_loss"], label="val_loss")
		plt.plot(history["train_acc"], label="train_acc")
		plt.plot(history["val_acc"], label="val_acc")

	plt.title("Training Loss and Accuracy on Dataset")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(figure_name)
	return


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

allData = torch.load('preprocessing\data_dict.pt')
all_images = allData["data"]
all_labels = allData["label"]
all_images = all_images.unsqueeze(1) # Add a new dimension with size 1
all_labels = all_labels.unsqueeze(1).unsqueeze(2).unsqueeze(3)
combined_tensor = torch.stack([all_images, all_labels], dim=1)
print(combined_tensor.shape)
TRAINDATA_SPLIT = 0.90
TESTDATA_SPLIT = 1 - TRAINDATA_SPLIT
print(len(allData))
numTraindataSamples = int(len(allData) * TRAINDATA_SPLIT)
numTestSamples = int(len(allData) * TESTDATA_SPLIT)
(trainData, testData) = random_split(allData,
	[numTraindataSamples, numTestSamples],
	generator=torch.Generator().manual_seed(42))

learning_rates = [0.00001,0.0001,0.001,0.01]
batch_sizes = [50,100,200,300,500]
num_epochs = [10,20,40,80]
loss_functions = [nn.NLLLoss()]
num_optm = 4

performance_history = pd.DataFrame(columns=[['model_num'],['batch_size'],['num_epoch'],['loss_function'],['accuracy'],['loss'],['training_time']])
count = 0
for learning_rate, batch_size, num_epoch, loss_function in itertools.product(learning_rates, batch_sizes, num_epochs, loss_functions):
	for optm in range(4):
		count +=1
		model, training_time, accuracy, history = one_iteration(learning_rate, batch_size, num_epoch, loss_function, optm, trainData, testData, device)
		# What to store on each model: model itself(With parameters), training/validation history and testing result
		torch.save(model, f"CNNModels/lr{learning_rate}bs{batch_size}ne{num_epoch}lf{loss_function}")
		with open(f"CNNModels/lr{learning_rate}bs{batch_size}ne{num_epoch}lf{loss_function}.json", 'w') as f:
			json.dump(history, f)














