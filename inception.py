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

df = pd.read_csv('BITVehicle/VehicleInfo.csv')

data = df.values

data = np.asarray(data)

crop_params = []
for i in data:
  crop_params.append({'top':i[5], 'bottom':i[7], 'left':i[4], 'right':i[6]})

class_names = ['SUV', 'Sedan', 'Microbus', 'Minivan', 'Truck', 'Bus']
class_name_to_index = {class_name: i for i, class_name in enumerate(class_names)}

y_indices = [class_name_to_index[class_name] for class_name in tuple(data[:,-2])]

y_one_hot = torch.nn.functional.one_hot(torch.tensor(y_indices), num_classes=len(class_names))

filepaths = tuple(data[:,1])

crop_params = tuple(crop_params)

y_values = y_one_hot
batch_size = 8

filepaths = list(filepaths)

for count,i in enumerate(filepaths):
  filepaths[count]='./BITVehicle/'+i

class CustomDataset(Dataset):
    def __init__(self, filepaths, crop_params_list, y_values, transform=None):
        self.filepaths = filepaths
        self.crop_params_list = crop_params_list
        self.y_values = y_values
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        crop_params = self.crop_params_list[idx]
        y = self.y_values[idx]


        image = Image.open(filepath).convert('RGB')
        image = self.crop_and_resize(image, crop_params)


        if self.transform:
            image = self.transform(image)

        return image, y

    def crop_and_resize(self, image, crop_params):

        image = transforms.functional.crop(image, crop_params['top'], crop_params['left'],
                                           crop_params['bottom'] - crop_params['top'],
                                           crop_params['right'] - crop_params['left'])
        image = transforms.functional.resize(image, (299, 299))
        return image

crop_params_list = list(crop_params)

transform = transforms.Compose([
    transforms.ToTensor(),])


dataset = CustomDataset(filepaths, crop_params_list, y_values, transform)


batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class CustomInceptionV3(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomInceptionV3, self).__init__()


        self.inception = models.inception_v3(pretrained=True)

        num_ftrs = self.inception.fc.in_features
        print(f"Number of input features (num_ftrs): {num_ftrs}")

        self.fc1 = nn.Linear(1000, 1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 256)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(256, num_classes)

    def forward(self, x, return_intermediate=False):

        inception_outputs = self.inception(x)
        x = inception_outputs

        #x = inception_outputs.logits


        x1 = self.fc1(x)
        x1 = self.relu1(x1)
        x1 = self.dropout1(x1)
        x2 = self.fc2(x1)
        x2 = self.relu2(x2)
        x2 = self.dropout2(x2)
        x3 = self.fc3(x2)
        x3 = self.relu3(x3)
        x3 = self.dropout3(x3)
        x4 = self.fc4(x3)

        if return_intermediate:
            return x4, x3, x2, x1, x
        else:
            return x4

model = CustomInceptionV3()

from tqdm import tqdm

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 15
for epoch in range(num_epochs):
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as t:
        for inputs, labels in t:

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)


            logits = outputs


            loss = criterion(logits, torch.argmax(labels, dim=1))


            loss.backward()
            optimizer.step()


            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct_predictions += (predicted == torch.argmax(labels, dim=1)).sum().item()
            total_samples += inputs.size(0)


            t.set_postfix(loss=total_loss / total_samples, accuracy=correct_predictions / total_samples)


    epoch_loss = total_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")


torch.save(model, 'trained_model_2.pth')