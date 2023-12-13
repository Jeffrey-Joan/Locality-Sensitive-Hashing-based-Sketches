#!/bin/bash

# Install Kaggle package
pip install kaggle

# Configure Kaggle API key permissions
chmod 600 /content/kaggle.json

# Create a directory for the dataset
mkdir /content/BITVehicle_data

# Download the dataset using Kaggle CLI
kaggle datasets download -d kuanghangdong/bitvehicle -p /content/BITVehicle_data

# Unzip the downloaded dataset
unzip /content/BITVehicle_data/bitvehicle.zip -d /content/BITVehicle_data
