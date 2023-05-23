import torch
import matplotlib.pyplot as plt
import numpy as np


device = "cpu"

learning_rate = "0.00100"
batch_size = "50"
num_epoch = "20"
loss_function = "NLLLoss()"
optm = "1"
layers = "3"
path = f"CNNModels\lr{learning_rate}bs{batch_size}ne{num_epoch}lf{loss_function}opt{optm}conv{layers}"
model = torch.load(path).to(device)

for i in range(3, 10+1):
    data_path = f"preprocessing\combining\combined_sensor{i}.pt"
    data = torch.load(data_path)[:, None, :, :].float().to(device)
    print(data.shape)

    result = model.forward(data)
    print(result)
    result_ = torch.split(result, 1, 1)
    result_0 = 10**result_[0].detach().numpy()
    result_1 = 10**result_[1].detach().numpy()

    comb = (np.add(result_1, -1 * result_0) + 1)/2
    result = 10**result.abs().detach().numpy()
    output = np.argmax(result, axis=1)
    plt.figure(figsize=(5, 10))
    plt.xlim((0, len(comb)))
    plt.xticks([0, len(comb)*3/10, len(comb)*5/10, len(comb)*7/10, len(comb)], [0, 3, 5, 7, 10])
    plt.yticks([0.0, 0.25, 0.50, 0.75, 1.0], ["0.00", "0.25", "0.50", "0.75", "1.00"])
    plt.grid(color='0.8', linestyle='-', linewidth=0.5)
    plt.plot(np.arange(len(comb)), comb)
    plt.ylabel("Probability [-]")
    plt.xlabel("Time [min]")
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    # fig.suptitle(f"Sensor {i}")
    # ax1.plot(np.arange(len(result_0)), result_0)
    # ax1.set_title("Prop Damaged")
    # ax2.plot(np.arange(len(result_1)), result_1)
    # ax2.set_title("Prop Healthy")
    # ax3.plot(np.arange(len(comb)), comb)
    # ax3.set_title("Comb prop Healthy")
    # ax4.plot(np.arange(len(output)), output)
    # ax4.set_title("Output value")
    plt.savefig(f"preprocessing\combining\sensor{i}_gen.png")






