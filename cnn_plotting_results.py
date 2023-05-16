import os
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'figure.max_open_warning': 0})
import seaborn as sns

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
                ne = re.search(r'ne(\d+)', filename[:-7]).group(1)
                print(ne)
                acc = history["test_results"][0]
                plt.plot(history["train_loss"], label="train_loss")
                plt.plot(history["val_loss"], label="validation_loss")
                plt.plot(history["train_acc"], label=f"train_accuracy")
                plt.plot(history["val_acc"], label=f"validation_accuracy")
                plt.plot(int(ne), acc, marker='X', markerfacecolor='black', ls='none', ms=10, label="test_accuracy")
                print(len(history["val_loss"]))
            plt.ylabel("Accuracy/Loss")
            plt.xlabel("Epochs")
            plt.legend(loc="center right")
            plt.ylim([0, 1.01])
            plt.title(f'Final model')
            plt.savefig(f'plots/plot_{num}')
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
                opt = re.search(r'opt(\d+)', filename[:-7]).group(1)
                conv = re.search(r'conv(\d)', filename[:-7]).group(1)
                tvt = loaded_data["time_taken"][0]
                tpc = loaded_data["time_taken"][1]
                num += 1
                ranking.append([num, acc, precs, recs, f1s, lr, bs, ne, lf, opt, conv, tvt, tpc])
    ranking = sorted(ranking, key=lambda x: x[0], reverse=True)
    ranking_df = pd.DataFrame(ranking, columns=['plot', 'accuracy', 'precisions', 'recalls', 'f1-scores',
                                                'learning rate', 'batch size', 'number epochs', 'loss function',
                                                'optimizer', 'convolutional layers', 'train-validation time',
                                                'time per computation'])
    ranking_df.index = ranking_df.index + 1
    ranking_df = ranking_df.set_index('plot')
    ranking_df = ranking_df.sort_index(ascending=True)
    print(ranking_df.to_string())
    ranking_df.to_csv('ranking_system.csv')
    return

def confusion_matrix(cf_matrix):
    group_names = ['True Positive', 'False Negative', 'False Positive', 'True Negative']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in (cf_matrix.flatten() / np.sum(cf_matrix))]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='coolwarm')
    plt.savefig('confusion_matrix.png')


confusion_matrix(np.array([[602,0],[2,524]]))
#graph_model_losses()
#ranking_system()

# Make sure the plots have the same range and domain, do one model per plot
#TN FP FN TP
#recall: 100%
#precision: 99.668874%
#F1 score: .998341625
#Test size: 1128

