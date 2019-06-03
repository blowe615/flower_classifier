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
