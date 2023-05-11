import torch
from cnn_architecture import newCNN3
import pickle
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

learning_rate = "0.00010"
batch_size = "50"
num_epoch = "20"
loss_function = "NLLLoss()"
optm = "2"
layers = "12"
path = f"CNNModels\lr{learning_rate}bs{batch_size}ne{num_epoch}lf{loss_function}opt{optm}conv{layers}"
model = torch.load(path).to(device)

data_path = "preprocessing\combining\combined_sensor3.pt"
data = torch.load(data_path)[:, None, :, :].float().to(device)
print(data.shape)

result = model.forward(data)
result_ = torch.split(result, 1, 1)
result_0 = result_[0].detach().numpy()
result_1 = result_[1].detach().numpy()

plt.figure()
plt.plot(np.arange(len(result_0)), result_0)
plt.show()
# plt.figure()
# plt.plot(np.arange(len(result_1)), result_1)
# plt.show()


