# Import packages
import numpy as np
from collections import OrderedDict

import json
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
#import matplotlib.pyplot as plt

from workspace_utils import active_session
from train_functions import load_dataset, define_transforms, label_mapping

# Load the data
data_directory = 'flower_data'
train_dir, valid_dir, test_dir = load_dataset(data_directory)

# Define transforms
train_transforms = define_transforms(train=True)
validate_transforms = define_transforms(train=False)
test_transforms = define_transforms(train=False)

# Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
validate_dataset = datasets.ImageFolder(valid_dir, transform=validate_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# Using the image datasets and the transforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle=True)
validloader = torch.utils.data.DataLoader(validate_dataset, batch_size = 32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle=True)

# Load dictionary mapping labels to category names
cat_to_name = label_mapping('cat_to_name.json')

print(cat_to_name['1'])
