import torch
import cnn_funcs_ as fn

device = "cpu"

SENSOR = False
MODEL_AMED = True

if MODEL_AMED:
    name = "AMED"
else:
    name = "FRED_30"


if not SENSOR:
    TRAINING = False
    if MODEL_AMED:
        train_dict = torch.load('train_data_dict.pt')
        val_dict = torch.load('val_data_dict.pt')
        test_dict = torch.load('test_data_dict.pt')
        if TRAINING:
            model = fn.train_ampl_model(train_dict, val_dict, test_dict).to(device)
        else:
            path = "CNNModels/final_amplitude_model"
            model = torch.load(path).to(device)
    else:
        train_dict = torch.load('train_data_dict_30.pt')
        val_dict = torch.load('val_data_dict_30.pt')
        test_dict = torch.load('test_data_dict_30.pt')
        if TRAINING:
            model = fn.train_freq_model(train_dict, val_dict, test_dict).to(device)
        else:
            path = "CNNFREQModels/lr0.00100bs50ne20lfNLLLoss()opt1conv3maxpsize3.4fclayers2"
            model = torch.load(path).to(device)

    noise_list = [[0, 200, 100, 50, 20, 10, 5, 2, 1], ["ZER0", "005", "01", "02", "05", "10", "20", "50", "100"]]
    noise_list = [[0], ["0"]]

    for i in range(len(noise_list[0])):
        new_data, f_list = fn.generate_data(test_dict["data"], test_dict["label"][:, 1], noise_list[0][i])
        new_data = new_data.float().to(device)
        output = fn.run_save_graph(model, new_data, f"{name}_{noise_list[1][i]}_noise", f_list)

else:
    sen_start = 3
    sen_end = 10
    val_split = 0.15
    sen_test_list = [3]

    sen_name = "3"

    test_dict = {"data": torch.Tensor([]).to(device), "label": torch.Tensor([]).to(device)}
    train_val_dict = {"data": torch.Tensor([]).to(device), "label": torch.Tensor([]).to(device)}
    for j in range(sen_start, sen_end+1):
        sen_dict = torch.load(f"data_dict/{name}_sensor_{j}.pt")
        if j in sen_test_list:
            test_dict["data"] = torch.cat((test_dict["data"], sen_dict["data"]), 0)
            test_dict["label"] = torch.cat((test_dict["label"], sen_dict["label"]), 0)
        else:
            train_val_dict["data"] = torch.cat((train_val_dict["data"], sen_dict["data"]), 0)
            train_val_dict["label"] = torch.cat((train_val_dict["label"], sen_dict["label"]), 0)

    train_dict, val_dict = fn.split_train_val_dict(train_val_dict, val_split)
    if MODEL_AMED:
        model = fn.train_ampl_model(train_dict, val_dict, test_dict).to(device)
    else:
        model = fn.train_freq_model(train_dict, val_dict, test_dict).to(device)

    noise_list = [[0, 200, 100, 50, 20, 10, 5, 2, 1], ["ZER0", "005", "01", "02", "05", "10", "20", "50", "100"]]

    for i in range(len(noise_list[0])):
        new_data, f_list = fn.generate_data(test_dict["data"], test_dict["label"][:, 1], noise_list[0][i])
        new_data = new_data.float().to(device)

        output = fn.run_save_graph(model, new_data, f"{name}_test_sensor_{sen_name}_{noise_list[1][i]}_noise", f_list)
