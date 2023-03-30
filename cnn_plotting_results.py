import os
import pickle
import re
import pandas as pd


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

def ranking_system():
	# assign directory
	directory = 'CNNModels'
	ranking = []


	for filename in os.listdir(directory):
		f = os.path.join(directory, filename)
		if "pickle" in str(f):
			with open(f, 'rb') as data:
				loaded_data = pickle.load(data)
				acc = loaded_data["test_results"][0]
				lr = re.search(r'lr(\d+\.\d+)', filename[:-7]).group(1)
				bs = re.search(r'bs(\d+)', filename[:-7]).group(1)
				ne = re.search(r'ne(\d+)', filename[:-7]).group(1)
				lf = re.search(r'lf(.+?)opt', filename[:-7]).group(1)
				opt = re.search(r'opt(\d)', filename[:-7]).group(1)
				ranking.append([acc, lr, bs, ne, lf, opt])
	ranking = sorted(ranking, key=lambda x: x[0], reverse=True)
	ranking_df = pd.DataFrame(ranking, columns = ['accuracy', 'learning rate', 'batch size', 'number epochs', 'loss function', 'optimizer'])
	ranking_df.index = ranking_df.index +1
	print(ranking_df)
	return

ranking_system()

1e-5