import pandas as pd
from torch.utils.data import Dataset
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader


class CreateDataset(Dataset):
	def __init__(self, img_path, img_labels, transform=None, target_transform=None):
		self.img_labels = img_labels
		self.img_path = img_path
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.img_labels)

	def __getitem__(self, idx):
		img_file = self.img_labels.iloc[idx, 0]
		image = torch.load(f"{self.img_path}/{img_file}").float()[None, :, :]
		label = self.img_labels.iloc[idx, 1]
		# if self.transform:
		# 	image = self.transform(image)
		# if self.target_transform:
		# 	label = self.target_transform(label)
		return image, label


if __name__ == '__main__':
	SPLIT_BY_SENSORS = False
	print("Split by sensor is", SPLIT_BY_SENSORS)
	if SPLIT_BY_SENSORS:
		TRAIN_SENSORS = [3, 4, 5, 6, 7]
		TRAIN_SPLIT = 0.65
		VAL_SPLIT = 1 - TRAIN_SPLIT
		TEST_SENSORS = [8, 9, 10]
	else:
		TRAIN_SPLIT = 0.65
		VAL_SPLIT = 0.20
		TEST_SPLIT = 0.15

	print("[INFO] Read image list")
	all_img_label_list = pd.read_csv("preprocessing/images/img_labels.cvs")

	print("[INFO] Split list")
	if SPLIT_BY_SENSORS:
		train_img_label_list = all_img_label_list[all_img_label_list['sensors'].isin(TRAIN_SENSORS)]
		test_img_label_list = all_img_label_list[all_img_label_list['sensors'].isin(TEST_SENSORS)]
		train_data = CreateDataset("preprocessing", train_img_label_list)
		test_data = CreateDataset("preprocessing", test_img_label_list)
		num_train = int(round(len(train_data) * TRAIN_SPLIT, 0))
		num_test = len(test_data)
		num_val = int(round(len(train_data) * VAL_SPLIT, 0))
		(train_data, val_data) = random_split(train_data, [num_train, num_val], generator=torch.Generator().manual_seed(42))
	else:
		all_data = CreateDataset("preprocessing", all_img_label_list)
		num_train = int(round(len(all_data) * TRAIN_SPLIT, 0))
		num_val = int(round(len(all_data) * VAL_SPLIT, 0))
		num_test = int(round(len(all_data) * TEST_SPLIT, 0))
		(train_data, val_data, test_data) = random_split(all_data,[num_train, num_val, num_test], generator=torch.Generator().manual_seed(42))

	print("[INFO] Load dataLoader")

	train_dataloader = DataLoader(train_data, batch_size=num_train, shuffle=True)
	val_dataloader = DataLoader(val_data, batch_size=num_val, shuffle=True)
	test_dataloader = DataLoader(test_data, batch_size=num_test, shuffle=True)

	print("[INFO] Save datasets (this takes a while because it has to open and read 7520 files)")

	train_features, train_labels = next(iter(train_dataloader))
	train_ind = torch.arange(0, len(train_features))
	train_labels = torch.stack((train_ind, train_labels), -1)
	train_data_dict = {"data": train_features, "label": train_labels}
	torch.save(train_data_dict, "train_data_dict.pt")
	print("[INFO] Saving training dataset complete")

	val_features, val_labels = next(iter(val_dataloader))
	val_ind = torch.arange(0, len(val_features))
	val_labels = torch.stack((val_ind, val_labels), -1)
	val_data_dict = {"data": val_features, "label": val_labels}
	torch.save(val_data_dict, "val_data_dict.pt")
	print("[INFO] Saving validating dataset complete")

	test_features, test_labels = next(iter(test_dataloader))
	test_ind = torch.arange(0, len(test_features))
	test_labels = torch.stack((test_ind, test_labels), -1)
	test_data_dict = {"data": test_features, "label": test_labels}
	torch.save(test_data_dict, "test_data_dict.pt")
	print("[INFO] Saving testing dataset complete")






