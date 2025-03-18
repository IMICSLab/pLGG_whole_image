import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
from scipy import stats
import scipy.stats as st
# import seaborn as sns
from scipy.stats import t
from math import sqrt
from statistics import stdev
from scipy.stats import t
from sklearn.metrics import confusion_matrix
from ast import literal_eval
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score

f1 = "plgg_whole_image_shallow_segFalse_results.csv"
f2 = "plgg_whole_image_shallow_segTrue_results_.csv"
df = pd.read_csv(f1)
df = df.drop(columns = ["Unnamed: 0"])
d2 = pd.read_csv(f2)
d2 = d2.drop(columns = ["Unnamed: 0"])



# Produce confusion matrix for each trial
confusion_matrices = []
overall_accuracies = []
overall_precisions = []
overall_recalls = []
per_class_precisions = []
per_class_recalls = []
overall_aucs = []
per_class_aucs = []
for i in range(len(d2)):
    print(i)
    true_class = literal_eval(d2["Class Test True Labels"][i])
    preds = literal_eval(d2["Class Test Estimated Labels"][i])
    preds = np.array([np.array(xi) for xi in preds])
    pred_class = np.argmax(preds, axis = 1)
    confusion_matrix_out = confusion_matrix(true_class, pred_class)
    confusion_matrices.append(confusion_matrix_out)
    overall_accuracies.append(accuracy_score(true_class, pred_class))
    overall_aucs.append(roc_auc_score(true_class, preds, multi_class="ovr", average="macro"))
    per_class_aucs.append(roc_auc_score(true_class, preds, multi_class="ovr", average=None))
    overall_precisions.append(precision_score(true_class, pred_class, average = "macro"))
    per_class_precisions.append(precision_score(true_class, pred_class, average=None))
    overall_recalls.append(recall_score(true_class, pred_class, average = "macro"))
    per_class_recalls.append(recall_score(true_class, pred_class, average=None))
print("overall CM:")
print(np.mean(confusion_matrices, axis =0))
print("overall accuracy:")
print(np.mean(overall_accuracies, axis=0))
print("total test cases:")
print(np.sum(np.mean(confusion_matrices, axis =0)))
print("overall proportions in CM:")
print(np.mean(confusion_matrices, axis =0)/np.sum(np.mean(confusion_matrices, axis =0)))
print("overall AUC")
print(np.mean(overall_aucs, axis =0))
print("per class AUC")
print(np.mean(per_class_aucs, axis =0))
print("overall precision")
print(np.mean(overall_precisions, axis =0))
print("per class  precision")
print(np.mean(per_class_precisions, axis =0))
print("overall recall")
print(np.mean(overall_recalls, axis =0))
print("perclass recall")
print(np.mean(per_class_recalls, axis =0))

