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
f1 = "plgg_whole_image_results.csv"
f2 = "no_brats_plgg_whole_image_results.csv"

df = pd.read_csv(f1)
df = df.drop(columns = ["Unnamed: 0"])
d2 = pd.read_csv(f2)
d2 = d2.drop(columns = ["Unnamed: 0"])


print("With brats")
print(f"Mean AUC: {np.mean(df['Class Test AUCs'].values)}, 95% CI for mean: "
      f"{st.norm.interval(confidence=0.95, loc=np.mean(df['Class Test AUCs'].values), scale=st.sem(df['Class Test AUCs'].values))}\n")
print("No brats")
print(f"Mean AUC: {np.mean(d2['Class Test AUCs'].values)}, 95% CI for mean: "
      f"{st.norm.interval(confidence=0.95, loc=np.mean(d2['Class Test AUCs'].values), scale=st.sem(d2['Class Test AUCs'].values))}\n")

# Calculate p-value
print("\n\nP value stuff: ")
#Statistical test: Combined vs Radiomics
diff = [ x-y for y, x in zip(df['Class Test AUCs'], d2['Class Test AUCs'])]
print(diff)
# for i in range(len(diff)):
#     print(i)
#     if diff[i]>0.05:
#         diff[i]=0.1
#Comopute the mean of differences
d_bar = np.mean(diff)
#compute the variance of differences
sigma2 = np.var(diff)
print(d_bar)
print(sigma2)
#compute the number of data points used for training
n1 = 0.9
#compute the number of data points used for testing
n2 = 0.1
#compute the total number of data points
n = len(diff)
#compute the modified variance
sigma2_mod = sigma2 * (1/n + n2/n1)
#compute the t_static
t_static =  d_bar / np.sqrt(sigma2_mod)
#Compute p-value and plot the results
Pvalue = ((1 - t.cdf(t_static, n-1)))
print(t_static)
print(Pvalue)
print('\n')
