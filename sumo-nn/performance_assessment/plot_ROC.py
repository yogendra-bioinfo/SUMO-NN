## This script plot the ROC curve, More details can be found on Stack Overflow here - https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python

import sklearn.metrics as metrics
import matplotlib.pyplot as plt

input_file = open("predictions_on_test_data.tsv", "r")
input_data = input_file.readlines()
input_file.close()
print(len(input_data))
print(input_data[0])
print(input_data[-1])

preds, y_test = [], []
for line in input_data[1:]:
    words = line.strip().split('\t')
    preds.append(int(words[0]))
    y_test.append(int(words[1]))
print(len(preds), len(y_test))

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
