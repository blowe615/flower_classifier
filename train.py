# Import packages
import numpy as np
from collections import OrderedDict

import json
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

from workspace_utils import active_session
from helper_functions import train_parser, load_dataset, define_transforms, label_mapping
from model_functions import DeepNetworkClassifier, train, test

# Parse arguments from command line
arg_dict = train_parser()

# Load the data
data_directory = arg_dict['data_dir']
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

# Create a dictionary of torchvision models to choose from
print('Downloading models...')
arch_dict = {'vgg19_bn': models.vgg19_bn(pretrained=True),
             'alexnet': models.alexnet(pretrained=True)}

# Create a dictionary of input sizes for each of the available torchvision models
arch_input_dict = {'vgg19_bn': 25088,
                   'alexnet': 9216}

arch=arg_dict['arch']

# Create an instance of the user-specified pretrained network
model = arch_dict[arch]

# Identify the device on which the model will be trained
GPU = arg_dict['gpu']
if GPU:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Turn off gradient calculations for model parameters
for param in model.parameters():
    param.requires_grad = False

# Build the model classifier
hidden_units = arg_dict['hidden_units']
classifier = DeepNetworkClassifier(arch_input_dict[arch], 102, hidden_units)

model.classifier = classifier

# Add the NLLLoss criterion (since LogSoftmax is called in the classifier)
criterion = nn.NLLLoss()

# Define the optimizer
learning_rate = arg_dict['learning_rate']
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# Send the model to the device
model.to(device);

# Train the network
epochs = arg_dict['epochs']

if device == torch.device('cuda'):
    # Keep the workspace active
    with active_session():
        print('Begin training...')
        train(model, trainloader, validloader, criterion, optimizer, epochs, device)
else:
    print('Begin training...')
    train(model, trainloader, validloader, criterion, optimizer, epochs, device)

# Test the network
test(model, testloader, device)

# Save the checkpoint
checkpoint = {'arch':arch,
              'input_size': arch_input_dict[arch],
              'output_size': 102,
              'hidden_layers': hidden_units,
              'state_dict': model.state_dict(),
              'epochs': epochs,
              'optimizer_state_dict': optimizer.state_dict(),
              'class_to_idx':train_dataset.class_to_idx}

save_dir=arg_dict['save_dir']
torch.save(checkpoint, save_dir)
print('Model saved to', save_dir)