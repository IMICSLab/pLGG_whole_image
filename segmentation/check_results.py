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


f1 = "medicalnet_plgg_results.csv"
f2 = "mednext_plgg_results.csv"
f3 = "transbts_plgg_results.csv"


df = pd.read_csv(f1)
df = df.drop(columns = ["Unnamed: 0"])
d2 = pd.read_csv(f2)
d2 = d2.drop(columns = ["Unnamed: 0"])
d3 = pd.read_csv(f3)
d3= d3.drop(columns = ["Unnamed: 0"])


print("CNN")
print(f"Mean : {np.mean(d2['Seg Test Dices'].values)}, 95% CI: "
      f"{st.norm.interval(confidence=0.95, loc=np.mean(d2['Seg Test Dices'].values), scale=st.sem(d2['Seg Test Dices'].values))}\n")

print("ConvNeXt")
print(f"Mean: {np.mean(df['Seg Test Dices'].values)}, 95% CI: "
      f"{st.norm.interval(confidence=0.95, loc=np.mean(df['Seg Test Dices'].values), scale=st.sem(df['Seg Test Dices'].values))}\n")

print("TransBTS")
print(f"Mean : {np.mean(d3['Seg Test Dices'].values)}, 95% CI: "
      f"{st.norm.interval(confidence=0.95, loc=np.mean(d3['Seg Test Dices'].values), scale=st.sem(d3['Seg Test Dices'].values))}\n")


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


#Statistical test: ConvNeXT vs TransBTS
diff = [ x-y for y, x in zip(d3['Seg Test Dices'], d2['Seg Test Dices'])]
d_bar = np.mean(diff)
sigma2 = np.var(diff)
n1 = 0.9
n2 = 0.1
n = len(diff)
sigma2_mod = sigma2 * (1/n + n2/n1)
t_static =  d_bar / np.sqrt(sigma2_mod)
Pvalue = ((1 - t.cdf(t_static, n-1)))
print(f'Pvalue ConvNext vs TransBTS {Pvalue}')
print('\n')

#Statistical test: CNN vs TransBTS
diff = [ x-y for y, x in zip(d3['Seg Test Dices'], df['Seg Test Dices'])]
d_bar = np.mean(diff)
sigma2 = np.var(diff)
n1 = 0.9
n2 = 0.1
n = len(diff)
sigma2_mod = sigma2 * (1/n + n2/n1)
t_static =  d_bar / np.sqrt(sigma2_mod)
Pvalue = ((1 - t.cdf(t_static, n-1)))
print(f'Pvalue CNN vs TransBTS {Pvalue}')
print('\n')

data1 = df['Seg Test Dices'].values
data2 = d2['Seg Test Dices'].values
data3 = d3['Seg Test Dices'].values

data = [data1, data2, data3]

labels = [ 'CNN', 'ConvNeXt', 'TransBTS']

fig, ax = plt.subplots()

# Create boxplot
ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                 meanline=True, meanprops={'color': 'red', 'linestyle': '--', 'linewidth': 2})

# Add title and labels
ax.set_xlabel('Architecture', fontsize=16)
ax.set_ylabel('Segmentation Dice Score', fontsize=16)

# Increase the font size of tick labels
ax.tick_params(axis='both', which='major', labelsize=12)

# Save the figure and show
plt.tight_layout()
plt.show()
