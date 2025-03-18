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


f1 = "mednext_plgg_results.csv"
f2 = "no_brats_mednext_plgg_results.csv"


df = pd.read_csv(f1)
df = df.drop(columns = ["Unnamed: 0"])
d2 = pd.read_csv(f2)
d2 = d2.drop(columns = ["Unnamed: 0"])





print("With Brats pretraining")
print(f"Mean: {np.mean(df['Seg Test Dices'].values)}, 95% CI: "
      f"{st.norm.interval(confidence=0.95, loc=np.mean(df['Seg Test Dices'].values), scale=st.sem(df['Seg Test Dices'].values))}\n")

print("Without Brats pretraining")
print(f"Mean : {np.mean(d2['Seg Test Dices'].values)}, 95% CI: "
      f"{st.norm.interval(confidence=0.95, loc=np.mean(d2['Seg Test Dices'].values), scale=st.sem(d2['Seg Test Dices'].values))}\n")




#Statistical test: ConvNeXT vs CNN
diff = [ x-y for y, x in zip(df['Seg Test Dices'], d2['Seg Test Dices'])]
d_bar = np.mean(diff)
sigma2 = np.var(diff)
n1 = 0.9
n2 = 0.1
n = len(diff)
sigma2_mod = sigma2 * (1/n + n2/n1)
t_static =  d_bar / np.sqrt(sigma2_mod)
Pvalue = ((1 - t.cdf(t_static, n-1)))
print(f'Pvalue ConvNext vs CNN {Pvalue}')
print('\n')

