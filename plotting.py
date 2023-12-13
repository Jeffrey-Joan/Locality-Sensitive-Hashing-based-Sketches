import seaborn as sns
import pickle

import pandas as pd

import matplotlib.pyplot as plt

import os

pkl_files = os.listdir()[1:-1]

def plot_hash_distributions(pkl_file,ax, log=True):
  with open(pkl_file, 'rb') as f:
    ht = pickle.load(f)
  df = pd.DataFrame(ht)
  df = df.fillna(0)
  for i in df.columns:
    sns.kdeplot(data= df[i].values, log_scale=log, ax=ax)
    ax.set_title(f'Hash Distribution for dim={pkl_file[-6:-4]} and nbits={pkl_file[-9:-7]}')

fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 9))
axes = axes.flatten()
for i,j in zip(pkl_files,axes):
  plot_hash_distributions(i,j)

plt.tight_layout()
plt.show()

def plot_table_distributions(pkl_file, ax):
  with open(pkl_file, 'rb') as f:
    ht = pickle.load(f)


  for count,i in enumerate(ht):
    sns.kdeplot(data= i.values(), log_scale=False, label = f'Table {count}', ax=ax, legend=False)
    ax.set_title(f'Distribution for dim={pkl_file[-6:-4]} and nbits={pkl_file[-9:-7]}')


fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 9))
axes = axes.flatten()
for i,j in zip(pkl_files,axes):
  plot_table_distributions(i,j)

plt.tight_layout()
plt.show()