# Import packages
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
#import matplotlib.pyplot as plt
import numpy as np
#import random
from workspace_utils import active_session
from collections import OrderedDict
import json
import argparse

def train_parser():
    '''
    Create the argument parser for the CLI
    '''
    parser = argparse.ArgumentParser(
        description = 'Parser for training ImageNet deep network on flower images')
    parser.add_argument('data_dir',
        help='set directory to train from')
    parser.add_argument('--save_dir', nargs='?', default='checkpoint.pth',
        help='set directory to save checkpoint')
    parser.add_argument('--arch', nargs='?', default='vgg19_bn', choices=['vgg19_bn','alexnet'],
        help='select the model architecture, options are "vgg19_bn" or "alexnet"')
    parser.add_argument('--learning_rate', nargs='?', default=0.001, type=float,
        help='set the learning rate')
    parser.add_argument('--hidden_units', nargs='?', default=1000, type=int,
        help='set the number of hidden units')
    parser.add_argument('--epochs', nargs='?', default=3, type=int,
        help='set the number of training epochs')
    parser.add_argument('--gpu', action='store_true',
        help='use GPU for training')
    args=parser.parse_args()
    arg_dict=vars(args)
    return arg_dict
    
def load_dataset(data_directory):
    '''
    Takes a data directory and splits it into training, validation, and testing directories
    '''
    train_dir = data_directory + '/train'
    valid_dir = data_directory + '/valid'
    test_dir = data_directory + '/test'
    return train_dir, valid_dir, test_dir

def define_transforms(train=True):
    '''
    Creates transforms in torchvision.
    If train=True, adds random flipping and cropping transforms before ToTensor() and Normalize()
    '''
    if train:
        data_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomResizedCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    else:
        data_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    return data_transforms


def label_mapping(filename):
    '''
    Loads a dictionary mapping category labels to category names
    filename: JSON object
    '''
    with open(filename, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name

def load_checkpoint(filepath):
    '''
    Rebuilds a trained torchvision model
    filepath: string, contains filepath where the trained model parameters are saved in a dictionary
    returns: a pytorch model, the optimizer, and the number of epochs
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = torch.load(filepath, map_location=device)
    model = models.vgg19_bn(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    # Build the model layers
model.classifier = nn.Sequential(model_layers)
    model.load_state_dict(model_dict['state_dict'])
    model.class_to_idx = model_dict['class_to_idx']
    optimizer = optim.Adam(model.classifier.parameters())
    optimizer.load_state_dict(model_dict['optimizer_state_dict'])
    epochs = model_dict['epochs']

    return model, optimizer, epochs
