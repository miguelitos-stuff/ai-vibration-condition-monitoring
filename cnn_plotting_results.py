import os
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt


def graph_model_losses():
	plt.clf()
	plt.style.use("ggplot")
	plt.figure()
	# assign directory
	directory = 'CNNModels'
	ranking = []
	for counter, filename in enumerate(os.listdir(directory)):
		f = os.path.join(directory, filename)
		if "pickle" in str(f):
			with open(f, 'rb') as data:
				history = pickle.load(data)
				#plt.plot(history["train_loss"], label="train_loss")
				#plt.plot(history["val_loss"], label="val_loss")
				plt.plot(history["train_acc"], label=f"model{counter}_train")
				plt.plot(history["val_acc"], label=f"model{counter}_valid")
	plt.title("Training/Validation Loss")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig('plottet')
	return

graph_model_losses()
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
				precs = loaded_data["test_results"][1]
				recs = loaded_data["test_results"][2]
				f1s = loaded_data["test_results"][3]
				lr = re.search(r'lr(\d+\.\d+)', filename[:-7]).group(1)
				bs = re.search(r'bs(\d+)', filename[:-7]).group(1)
				ne = re.search(r'ne(\d+)', filename[:-7]).group(1)
				lf = re.search(r'lf(.+?)opt', filename[:-7]).group(1)
				opt = re.search(r'opt(\d)', filename[:-7]).group(1)
				ranking.append([acc, precs, recs, f1s, lr, bs, ne, lf, opt])
	ranking = sorted(ranking, key=lambda x: x[0], reverse=True)
	ranking_df = pd.DataFrame(ranking, columns = ['accuracy', 'precisions', 'recalls', 'f1-scores', 'learning rate', 'batch size', 'number epochs', 'loss function', 'optimizer'])
	ranking_df.index = ranking_df.index +1
	print(ranking_df.to_string())
	return


# Make sure the plots have the same range and domain, do one model per plot