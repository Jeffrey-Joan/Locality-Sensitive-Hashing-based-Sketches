from statistics import median
import pickle
import time
import seaborn as sns
from torchsummary import summary
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
os.environ['KAGGLE_CONFIG_DIR'] = '/content'
import random
random.seed(42)
import cv2
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models



ht_2_32 = LSH_sketch(nbits = 2, dim = 32, vec_size=len(v1))
start = time.time()
for batch in dataloader:

    batch_images, batch_labels = batch
    batch_images = batch_images.to('cuda')
    batch_labels = batch_labels.to('cuda')
    output = m(batch_images, return_intermediate=True)
    for i in output[3].cpu().detach().numpy():
      ht_2_32.insert(i)
end = time.time()
print('Total time taken for insertion', end-start)
results = []
start = time.time()
for batch in dataloader:

    batch_images, batch_labels = batch
    batch_images = batch_images.to('cuda')
    batch_labels = batch_labels.to('cuda')
    output = m(batch_images, return_intermediate=True)
    for j in output[3].cpu().detach().numpy():
      results.append(ht_2_32.lookup(j))
    break
end = time.time()
print(f'\n\nTotal time taken for querying:', end-start, '\n')
for count,i in enumerate(results):
    print('Label: ', batch_labels[count])
    print('Max:',max(i), ' Min:',min(i), ' Median:', median(i),'\n')

ht_4_32 = LSH_sketch(nbits = 4, dim = 32, vec_size=len(v1))
start = time.time()
for batch in dataloader:

    batch_images, batch_labels = batch
    batch_images = batch_images.to('cuda')
    batch_labels = batch_labels.to('cuda')
    output = m(batch_images, return_intermediate=True)
    for i in output[3].cpu().detach().numpy():
      ht_4_32.insert(i)
end = time.time()
print('Total time taken for insertion', end-start)
results = []
start = time.time()
for batch in dataloader:

    batch_images, batch_labels = batch
    batch_images = batch_images.to('cuda')
    batch_labels = batch_labels.to('cuda')
    output = m(batch_images, return_intermediate=True)
    for j in output[3].cpu().detach().numpy():
      results.append(ht_4_32.lookup(j))
    break
end = time.time()
print(f'\n\nTotal time taken for querying:', end-start, '\n')
for count,i in enumerate(results):
    print('Label: ', batch_labels[count])
    print('Max:',max(i), ' Min:',min(i), ' Median:', median(i),'\n')

ht_2_16 = LSH_sketch(nbits = 2, dim = 16, vec_size=len(v1))
start = time.time()
for batch in dataloader:

    batch_images, batch_labels = batch
    batch_images = batch_images.to('cuda')
    batch_labels = batch_labels.to('cuda')
    output = m(batch_images, return_intermediate=True)
    for i in output[3].cpu().detach().numpy():
      ht_2_16.insert(i)
end = time.time()
print('Total time taken for insertion', end-start)
results = []
start = time.time()
for batch in dataloader:

    batch_images, batch_labels = batch
    batch_images = batch_images.to('cuda')
    batch_labels = batch_labels.to('cuda')
    output = m(batch_images, return_intermediate=True)
    for j in output[3].cpu().detach().numpy():
      results.append(ht_2_16.lookup(j))
    break
end = time.time()
print(f'\n\nTotal time taken for querying:', end-start, '\n')
for count,i in enumerate(results):
    print('Label: ', batch_labels[count])
    print('Max:',max(i), ' Min:',min(i), ' Median:', median(i),'\n')

ht_4_16 = LSH_sketch(nbits = 4, dim = 16, vec_size=len(v1))
start = time.time()
for batch in dataloader:

    batch_images, batch_labels = batch
    batch_images = batch_images.to('cuda')
    batch_labels = batch_labels.to('cuda')
    output = m(batch_images, return_intermediate=True)
    for i in output[3].cpu().detach().numpy():
      ht_4_16.insert(i)
end = time.time()
print('Total time taken for insertion', end-start)
results = []
start = time.time()
for batch in dataloader:

    batch_images, batch_labels = batch
    batch_images = batch_images.to('cuda')
    batch_labels = batch_labels.to('cuda')
    output = m(batch_images, return_intermediate=True)
    for j in output[3].cpu().detach().numpy():
      results.append(ht_4_16.lookup(j))
    break
end = time.time()
print(f'\n\nTotal time taken for querying:', end-start, '\n')
for count,i in enumerate(results):
    print('Label: ', batch_labels[count])
    print('Max:',max(i), ' Min:',min(i), ' Median:', median(i),'\n')