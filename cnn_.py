import torch
import cnn_funcs_ as fn
import cnn_architecture as arc

device = "cpu"

train_dict = torch.load('train_spec_dict.pt')
val_dict = torch.load('val_spec_dict.pt')
test_dict = torch.load('test_spec_dict.pt')

model = fn.train_freq_model(train_dict, val_dict, test_dict).to(device)

new_data = fn.generate_data(test_dict["data"], test_dict["label"][:, 1]).float().to(device)
output = fn.run_save_graph(model, new_data, "FRED")
