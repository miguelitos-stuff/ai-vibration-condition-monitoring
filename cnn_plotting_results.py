import os
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt


def graph_model_losses():
	# assign directory
	directory = 'CNNModels'
	ranking = []
	num = 0
	for filename in os.listdir(directory):
		f = os.path.join(directory, filename)
		if "pickle" in str(f):
			num += 1
			plt.clf()
			plt.style.use("ggplot")
			plt.figure()
			with open(f, 'rb') as data:
				history = pickle.load(data)
				plt.plot(history["train_loss"], label="train")
				plt.plot(history["val_loss"], label="validation")
			plt.ylabel("Loss")
			plt.xlabel("Epochs")
			plt.legend(loc="lower left")
			plt.ylim([0, 1])
			plt.savefig(f'plot_loss_{num}')

			plt.clf()
			plt.style.use("ggplot")
			plt.figure()
			f = os.path.join(directory, filename)
			with open(f, 'rb') as data:
				history = pickle.load(data)
				plt.plot(history["train_acc"], label=f"train")
				plt.plot(history["val_acc"], label=f"validation")
			plt.ylabel("Accuracy")
			plt.xlabel("Epochs")
			plt.legend(loc="lower left")
			plt.ylim([0, 1])
			plt.savefig(f'plot_accuracy_{num}')
	return
def ranking_system():
	# assign directory
	directory = 'CNNModels'
	ranking = []
	num = 0
	for filename in os.listdir(directory):
		f = os.path.join(directory, filename)
		if "pickle" in str(f):
			with open(f, 'rb') as data:
				loaded_data = pickle.load(data)
				acc = loaded_data["test_results"][0]
				precs = loaded_data["test_results"][1]
				recs = loaded_data["test_results"][2]
				f1s = loaded_data["test_results"][3]
				lr = re.search(r'lr(\d+\.\d+)', filename[:-7]).group(1)
				bs = re.search(r'bs(\d+)', filename[:-7]).group(1)
				ne = re.search(r'ne(\d+)', filename[:-7]).group(1)
				lf = re.search(r'lf(.+?)opt', filename[:-7]).group(1)
				opt = re.search(r'opt(\d)', filename[:-7]).group(1)
				num += 1
				ranking.append([num, acc, precs, recs, f1s, lr, bs, ne, lf, opt, '4'])
	ranking = sorted(ranking, key=lambda x: x[0], reverse=True)
	ranking_df = pd.DataFrame(ranking, columns = ['plot', 'accuracy', 'precisions', 'recalls', 'f1-scores', 'learning rate', 'batch size', 'number epochs', 'loss function', 'optimizer', 'convolution layers'])
	ranking_df.index = ranking_df.index +1
	ranking_df = ranking_df.set_index('plot')
	ranking_df = ranking_df.sort_index(ascending=True)
	print(ranking_df.to_string())
	return
#graph_model_losses()
ranking_system()

# Make sure the plots have the same range and domain, do one model per plot

# Create 15 models which are the best, have an execel file which assigns their number with their properties and their performance