import torch
import cnn_funcs_ as fn

device = "cpu"

SENSOR = False
MODEL_AMED = True


if MODEL_AMED:
    name = "AMED"
    title = "AMED Augmented Data Output with Noise"
else:
    name = "FRED_30"
    title = "FRED Augmented Data Output with Noise"


if not SENSOR:
    TRAINING = False
    if MODEL_AMED:
        train_dict = torch.load('train_data_dict.pt')
        val_dict = torch.load('val_data_dict.pt')
        test_dict = torch.load('test_data_dict_noise_005.pt')
        if TRAINING:
            model = fn.train_ampl_model(train_dict, val_dict, test_dict).to(device)
        else:
            path = "CNNModels/final_amplitude_model"
            model = torch.load(path).to(device)
    else:
        train_dict = torch.load('train_data_dict_30.pt')
        val_dict = torch.load('val_data_dict_30.pt')
        test_dict = torch.load('test_data_dict_30_noise_005.pt')
        if TRAINING:
            model = fn.train_freq_model(train_dict, val_dict, test_dict).to(device)
        else:
            path = "CNNFREQModels/lr0.00100bs50ne20lfNLLLoss()opt1conv3maxpsize3.4fclayers2"
            model = torch.load(path).to(device)

    noise_list = [[0, 200, 100, 50, 20, 10, 5, 2, 1], ["ZER0", "005", "01", "02", "05", "10", "20", "50", "100"]]
    noise_list = [[0], ["_0"]]

    for i in range(len(noise_list[0])):
        new_data, f_list = fn.generate_data(test_dict["data"], test_dict["label"][:, 1], noise_list[0][i])
        new_data = new_data.float().to(device)
        output = fn.run_save_graph(model, new_data, f"{name}_std_005_noise_", f_list, title)

else:
    sen_start = 3
    sen_end = 10
    val_split = 0.15
    for sen_test in range(sen_start, sen_end+1):
        test_dict = {"data": torch.Tensor([]).to(device), "label": torch.Tensor([]).to(device)}
        train_val_dict = {"data": torch.Tensor([]).to(device), "label": torch.Tensor([]).to(device)}
        for j in range(sen_start, sen_end+1):
            sen_dict = torch.load(f"data_dict/{name}_sensor_{j}.pt")
            if j == sen_test:
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

        # # noise_list = [[0, 200, 100, 50, 20, 10, 5, 2, 1], ["ZER0", "005", "01", "02", "05", "10", "20", "50", "100"]]
        # noise_list = [[0], ["0"]]
        #
        # for i in range(len(noise_list[0])):
        #     new_data, f_list = fn.generate_data(test_dict["data"], test_dict["label"][:, 1], noise_list[0][i])
        #     new_data = new_data.float().to(device)
        #
        #     output = fn.run_save_graph(model, new_data, f"{name}_test_sensor_{sen_test}_{noise_list[1][i]}_noise", f_list, title)

        print("Finished with test sensor = ", sen_test)
