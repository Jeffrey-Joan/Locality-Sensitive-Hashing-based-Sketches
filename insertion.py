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

from inception import CustomDataset, CustomInceptionV3, dataloader, dataset, filepaths, crop_params_list, y_values, transform
from LSH_sketch import LSH_sketch
m = torch.load('./trained_model_2.pth', map_location=torch.device('cuda'))


ht_16_32 = LSH_sketch(nbits=16, dim =32, vec_size=len(v1))
start = time.time()
for batch in dataloader:

    batch_images, batch_labels = batch
    batch_images = batch_images.to('cuda')
    batch_labels = batch_labels.to('cuda')
    output = m(batch_images, return_intermediate=True)
    for i in output[3].cpu().detach().numpy():
      ht_16_32.insert(i)
end = time.time()
print('Total time taken', end-start)

with open('ht_16_32.pkl', 'wb') as f:
    pickle.dump(ht_16_32.tables, f)


ht_32_32 = LSH_sketch(nbits=32, dim =32, vec_size=len(v1))
start = time.time()
for batch in dataloader:

    batch_images, batch_labels = batch
    batch_images = batch_images.to('cuda')
    batch_labels = batch_labels.to('cuda')
    output = m(batch_images, return_intermediate=True)
    for i in output[3].cpu().detach().numpy():
      ht_32_32.insert(i)
end = time.time()
print('Total time taken', end-start)

with open('ht_32_32.pkl', 'wb') as f:
    pickle.dump(ht_32_32.tables, f)



ht_32_64 = LSH_sketch(nbits=32, dim =64, vec_size=len(v1))
start = time.time()
for batch in dataloader:

    batch_images, batch_labels = batch
    batch_images = batch_images.to('cuda')
    batch_labels = batch_labels.to('cuda')
    output = m(batch_images, return_intermediate=True)
    for i in output[3].cpu().detach().numpy():
      ht_32_64.insert(i)
end = time.time()
print('Total time taken', end-start)

with open('ht_32_64.pkl', 'wb') as f:
    pickle.dump(ht_32_64.tables, f)



ht_64_64 = LSH_sketch(nbits=64, dim =64, vec_size=len(v1))
start = time.time()
for batch in dataloader:

    batch_images, batch_labels = batch
    batch_images = batch_images.to('cuda')
    batch_labels = batch_labels.to('cuda')
    output = m(batch_images, return_intermediate=True)
    for i in output[3].cpu().detach().numpy():
      ht_64_64.insert(i)
end = time.time()
print('Total time taken', end-start)

with open('ht_64_64.pkl', 'wb') as f:
    pickle.dump(ht_64_64.tables, f)



ht_64_128 = LSH_sketch(nbits=64, dim =128, vec_size=len(v1))
start = time.time()
for batch in dataloader:

    batch_images, batch_labels = batch
    batch_images = batch_images.to('cuda')
    batch_labels = batch_labels.to('cuda')
    output = m(batch_images, return_intermediate=True)
    for i in output[3].cpu().detach().numpy():
      ht_64_128.insert(i)
end = time.time()
print('Total time taken', end-start)

with open('ht_64_128.pkl', 'wb') as f:
    pickle.dump(ht_64_128.tables, f)



ht_128_128 = LSH_sketch(nbits=128, dim =128, vec_size=len(v1))
start = time.time()
for batch in dataloader:

    batch_images, batch_labels = batch
    batch_images = batch_images.to('cuda')
    batch_labels = batch_labels.to('cuda')
    output = m(batch_images, return_intermediate=True)
    for i in output[3].cpu().detach().numpy():
      ht_128_128.insert(i)
end = time.time()
print('Total time taken', end-start)

with open('ht_128_128.pkl', 'wb') as f:
    pickle.dump(ht_128_128.tables, f)



ht_16_4 = LSH_sketch(nbits=16, dim =4, vec_size=1024)
start = time.time()
for batch in dataloader:

    batch_images, batch_labels = batch
    batch_images = batch_images.to('cuda')
    batch_labels = batch_labels.to('cuda')
    output = m(batch_images, return_intermediate=True)
    for i in output[3].cpu().detach().numpy():
      ht_16_4.insert(i)
end = time.time()
print('Total time taken', end-start)

with open('ht_16_4.pkl', 'wb') as f:
    pickle.dump(ht_16_4.tables, f)



ht_32_4 = LSH_sketch(nbits=32, dim =4, vec_size=1024)
start = time.time()
for batch in dataloader:

    batch_images, batch_labels = batch
    batch_images = batch_images.to('cuda')
    batch_labels = batch_labels.to('cuda')
    output = m(batch_images, return_intermediate=True)
    for i in output[3].cpu().detach().numpy():
      ht_32_4.insert(i)
end = time.time()
print('Total time taken', end-start)

with open('ht_32_4.pkl', 'wb') as f:
    pickle.dump(ht_32_4.tables, f)



ht_32_8 = LSH_sketch(nbits=32, dim =8, vec_size=1024)
start = time.time()
for batch in dataloader:

    batch_images, batch_labels = batch
    batch_images = batch_images.to('cuda')
    batch_labels = batch_labels.to('cuda')
    output = m(batch_images, return_intermediate=True)
    for i in output[3].cpu().detach().numpy():
      ht_32_8.insert(i)
end = time.time()
print('Total time taken', end-start)

with open('ht_32_8.pkl', 'wb') as f:
    pickle.dump(ht_32_8.tables, f)



ht_16_8 = LSH_sketch(nbits=16, dim =8, vec_size=1024)
start = time.time()
for batch in dataloader:

    batch_images, batch_labels = batch
    batch_images = batch_images.to('cuda')
    batch_labels = batch_labels.to('cuda')
    output = m(batch_images, return_intermediate=True)
    for i in output[3].cpu().detach().numpy():
      ht_16_8.insert(i)
end = time.time()
print('Total time taken', end-start)

with open('ht_16_8.pkl', 'wb') as f:
    pickle.dump(ht_16_8.tables, f)



ht_2_16 = LSH_sketch(nbits=2, dim =16, vec_size=1024)
start = time.time()
for batch in dataloader:

    batch_images, batch_labels = batch
    batch_images = batch_images.to('cuda')
    batch_labels = batch_labels.to('cuda')
    output = m(batch_images, return_intermediate=True)
    for i in output[3].cpu().detach().numpy():
      ht_2_16.insert(i)
end = time.time()
print('Total time taken', end-start)

with open('ht_2_16.pkl', 'wb') as f:
    pickle.dump(ht_2_16.tables, f)



ht_2_32 = LSH_sketch(nbits=2, dim =32, vec_size=1024)
start = time.time()
for batch in dataloader:

    batch_images, batch_labels = batch
    batch_images = batch_images.to('cuda')
    batch_labels = batch_labels.to('cuda')
    output = m(batch_images, return_intermediate=True)
    for i in output[3].cpu().detach().numpy():
      ht_2_32.insert(i)
end = time.time()
print('Total time taken', end-start)

with open('ht_2_32.pkl', 'wb') as f:
    pickle.dump(ht_2_32.tables, f)



ht_4_32 = LSH_sketch(nbits=4, dim =32, vec_size=1024)
start = time.time()
for batch in dataloader:

    batch_images, batch_labels = batch
    batch_images = batch_images.to('cuda')
    batch_labels = batch_labels.to('cuda')
    output = m(batch_images, return_intermediate=True)
    for i in output[3].cpu().detach().numpy():
      ht_4_32.insert(i)
end = time.time()
print('Total time taken', end-start)

with open('ht_4_32.pkl', 'wb') as f:
    pickle.dump(ht_4_32.tables, f)

