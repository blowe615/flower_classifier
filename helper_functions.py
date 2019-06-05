# Import packages
import numpy as np
from PIL import Image

import argparse
import json
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

from workspace_utils import active_session
from model_functions import DeepNetworkClassifier


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
    # Create a dictionary of torchvision models to choose from
    print('Downloading models...')
    arch_dict = {'vgg19_bn': models.vgg19_bn(pretrained=True),
                 'alexnet': models.alexnet(pretrained=True)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = torch.load(filepath, map_location=device)
    model = arch_dict[model_dict['arch']]
    for param in model.parameters():
        param.requires_grad = False
    # Build the model layers
    model.classifier = DeepNetworkClassifier(model_dict['input_size'], model_dict['output_size'], model_dict['hidden_layers'])
    model.load_state_dict(model_dict['state_dict'])
    model.class_to_idx = model_dict['class_to_idx']
    optimizer = optim.Adam(model.classifier.parameters())
    optimizer.load_state_dict(model_dict['optimizer_state_dict'])
    epochs = model_dict['epochs']

    return model, optimizer, epochs

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    # Open the image
    im = Image.open(image)
    # Obtain the dimensions of the image
    (w,h) = im.size
    # Resize the image to 256 pixels on the shortest side
    if w < h:
        aspect_ratio = h/w
        im.thumbnail((256, 256*aspect_ratio))
    else:
        aspect_ratio = w/h
        im.thumbnail((256*aspect_ratio, 256))
    # Obtain the new dimensions of the image
    (w_new,h_new) = im.size
    # Center crop the image to 224 x 224
    cropped_image = im.crop((w_new//2 - 112, h_new//2 - 112, w_new//2 + 112, h_new//2 + 112))
    # Convert image to numpy array
    np_im = np.array(cropped_image)
    # Scale from 0-255 to 0-1
    np_im = np_im / 255
    # Normalize
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image = (np_im - means) / stds
    # Rearrange dimensions to match expected input into PyTorch (color channel in 1st dim instead of 3rd dim)
    image = image.transpose((2,0,1))
    return image
