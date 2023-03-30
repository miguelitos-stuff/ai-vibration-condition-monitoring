# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
#from outdated_scripts.cnn_architecture2 import LeNet
from cnn_architecture import CNN
import cnn_architecture as arc
#from preprocessing import 'data_dict.pt'
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import random_split
from torch.utils.data import DataLoader
#from torchvision.transforms import ToTensor
#from torchvision.datasets import KMNIST
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
import pickle
import torch
import time
import os


def one_iteration(INIT_LR, BATCH_SIZE, EPOCHS, lossFn, optm, trainData, testData, device):
	# define the train and val splits
	TRAIN_SPLIT = 0.75
	VAL_SPLIT = 1 - TRAIN_SPLIT
	# set the device we will be using to train the model
	print("Pytorch CUDA Version is available:", torch.cuda.is_available())

	## load the KMNIST dataset
	#print("[INFO] loading the dataset...")
	#trainData = KMNIST(root="data", train=True, download=True,
	#	transform=ToTensor())
	#testData = KMNIST(root="data", train=False, download=True,
	#	transform=ToTensor())

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

	# initialize the CNN model
	print("[INFO] initializing the CNN model...")
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
	#elif optm == 2:
	#	opt = LBFGS(model.parameters(),lr=learning_rate)
	elif optm == 2 or optm == 3:
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
			y=y.type(torch.LongTensor)
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
				y=y.type(torch.LongTensor)
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
		targets = []
		# loop over the test set
		for (x, y) in testDataLoader:
			# send the input to the device
			x = x.to(device)
			y=y.type(torch.LongTensor)
			y=y.to(device)
			# make the predictions and add them to the list
			pred = model(x)
			preds.extend(pred.argmax(axis=1).cpu().numpy())
			totalTestLoss += lossFn(pred, y)
			testCorrect += (pred.argmax(1) == y).type(
				torch.float).sum().item()
			targets.extend(y.cpu().numpy())
		test_acc = testCorrect / len(testDataLoader.dataset)
		testSteps = len(testDataLoader.dataset)
		avgTestLoss = totalTestLoss / testSteps
	# generate a classification report
	print(f"Test loss: {avgTestLoss}, Test accuracy: {test_acc}")
	test_results = classification_report(targets,np.array(preds), target_names=[str(i) for i in range(2)])
	print(test_results)
	precision, recall, fscore, support = precision_recall_fscore_support(targets, np.array(preds))
	print(precision, recall, fscore, support)
	test_results = [test_acc, precision, recall, fscore, support]
	H["test_results"] = test_results
	return model, (endTime - startTime), H

def transform_lr(num):
	# Find the exponent
	exp = int(abs(num) // 1)
	if exp == 0:
		exp_str = ''
	else:
		exp_str = f'*1e-{exp:02d}'

	# Find the coefficient
	coeff = abs(num) / (10 ** exp)
	coeff_str = f'{coeff:.5f}'

	# Combine the coefficient and exponent strings
	if num < 0:
		output_str = '-' + coeff_str + exp_str
	else:
		output_str = coeff_str + exp_str
	return output_str

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

allData = torch.load('preprocessing\data_dict.pt')
all_images = allData["data"].float()[:,None,:,:]
all_labels = allData["label"][:, None]
all_idx = torch.arange(len(all_images)).to("cuda")[:, None]
all_labels = torch.cat((all_idx, all_labels), 1)
all_data = arc.CreateDataset(all_labels, all_images)

TRAINDATA_SPLIT = 0.90
TESTDATA_SPLIT = 1 - TRAINDATA_SPLIT
numTraindataSamples = int(round(len(all_data) * TRAINDATA_SPLIT, 0))
numTestSamples = int(round(len(all_data) * TESTDATA_SPLIT, 0))
(train_data, test_data) = random_split(all_data,
	[numTraindataSamples, numTestSamples],
	generator=torch.Generator().manual_seed(42))

trainData = train_data
testData = test_data

learning_rates = [0.00001,0.0001,0.001,0.01]
batch_sizes = [50]
num_epochs = [2,20]
loss_functions = [nn.NLLLoss()]
num_optm = 4

performance_history = pd.DataFrame(columns=[['model_num'],['batch_size'],['num_epoch'],['loss_function'],['accuracy'],['loss'],['training_time']])
count = 0
for learning_rate, batch_size, num_epoch, loss_function in itertools.product(learning_rates, batch_sizes, num_epochs, loss_functions):
	for optm in range(num_optm):
		print(f"Learning rate: {learning_rate}, Batch size: {batch_size}, Number epochs: {num_epoch}, Loss function{loss_function}, Optimizer: {optm}")
		count +=1
		model, training_time, history = one_iteration(learning_rate, batch_size, num_epoch, loss_function, optm, train_data, test_data, device)
		# What to store on each model: model itself(With parameters), training/validation history and testing result
		torch.save(model, f"CNNModels/lr{transform_lr(learning_rate)}bs{batch_size}ne{num_epoch}lf{loss_function}opt{optm}")
		with open(f"CNNModels/lr{transform_lr(learning_rate)}bs{batch_size}ne{num_epoch}lf{loss_function}opt{optm}.pickle", 'wb') as f:
			pickle.dump(history, f)














