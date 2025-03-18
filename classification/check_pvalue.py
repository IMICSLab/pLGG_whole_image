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
f1 = "plgg_whole_image_segFalse.csv"
f2 = "plgg_whole_image_segTrue.csv"
f3 = "plgg_manual_segmentation.csv"

df = pd.read_csv(f1)
df = df.drop(columns = ["Unnamed: 0"])
d2 = pd.read_csv(f2)
d2 = d2.drop(columns = ["Unnamed: 0"])
d3 = pd.read_csv(f3)
d3 = d3.drop(columns = ["Unnamed: 0"])

print("Without pretraining")
print(f"Mean AUC: {np.mean(df['Class Test AUCs'].values)}, 95% CI for mean: "
      f"{st.norm.interval(confidence=0.95, loc=np.mean(df['Class Test AUCs'].values), scale=st.sem(df['Class Test AUCs'].values))}\n")
print("With pretraining")
print(f"Mean AUC: {np.mean(d2['Class Test AUCs'].values)}, 95% CI for mean: "
      f"{st.norm.interval(confidence=0.95, loc=np.mean(d2['Class Test AUCs'].values), scale=st.sem(d2['Class Test AUCs'].values))}\n")
print("Manual segmentation")
print(f"Mean AUC: {np.mean(d3['Class Test AUCs'].values)}, 95% CI for mean: "
      f"{st.norm.interval(confidence=0.95, loc=np.mean(d3['Class Test AUCs'].values), scale=st.sem(d3['Class Test AUCs'].values))}\n")



#Statistical test: With and without pretraining
diff = [y - x for y, x in zip(d2['Class Test AUCs'], df['Class Test AUCs'])]
d_bar = np.mean(diff)
sigma2 = np.var(diff)
n1 = 0.9
n2 = 0.1
n = len(diff)
sigma2_mod = sigma2 * (1/n + n2/n1)
t_static =  d_bar / np.sqrt(sigma2_mod)
Pvalue = ((1 - t.cdf(t_static, n-1)))
print("Statistical test: With and without pretraining")
print(Pvalue)
print('\n')

#Statistical test: Manual segmentation vs without pretraining
diff = [y - x for y, x in zip(d3['Class Test AUCs'], df['Class Test AUCs'])]
d_bar = np.mean(diff)
sigma2 = np.var(diff)
n1 = 0.9
n2 = 0.1
n = len(diff)
sigma2_mod = sigma2 * (1/n + n2/n1)
t_static =  d_bar / np.sqrt(sigma2_mod)
Pvalue = ((1 - t.cdf(t_static, n-1)))
print("Statistical test: Manual and without pretraining")
print(Pvalue)
print('\n')


#Statistical test: With pretraining vs manual segmentation
diff = [y - x for y, x in zip(d2['Class Test AUCs'], d3['Class Test AUCs'])]
d_bar = np.mean(diff)
sigma2 = np.var(diff)
n1 = 0.9
n2 = 0.1
n = len(diff)
sigma2_mod = sigma2 * (1/n + n2/n1)
t_static =  d_bar / np.sqrt(sigma2_mod)
Pvalue = ((1 - t.cdf(t_static, n-1)))
print("Statistical test: With pretraining vs manual segmentation")
print(Pvalue)
print('\n')
