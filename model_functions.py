import torch
from torch import nn
import torch.nn.functional as F

arch_input_dict = {'vgg19_bn': 25088,
                   'alexnet': 9216}
arch='vgg19_bn'
arch_input_dict[arch]
102

class DeepNetworkClassifier(nn.Module):
    def __init__(self, input_units, output_units, hidden_units=[1000],p_drop=0.2):
        '''
        Builds a classifier for a pretrained deep neural network for the flower dataset

        Inputs
        ------
        arch: string, model name from torchvision.models, determines the number of inputs in the classifier
        hidden_units: list, the number of hidden layers and units in each hidden layer
        '''
        super().__init__()
        # Create input layer with input units based on model architecture
        self.layers = nn.ModuleList(nn.Linear(input_units,hidden_units[0])
        # Add hidden layers, if any
        for j in zip(hidden_units[:-1],hidden_units[1:]):
            self.layers.extend(nn.Linear(j))
        # Create output layer with 102 outputs (for 102 flower classes)
        self.output = nn.Linear(hidden_units[-1],output_units)
        # Define level of dropout
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, x):
        '''
        Performs a forward pass through the network and returns the log probabilities

        x: layer in model
        '''
        # Apply ReLU activation function and dropout to each layer before output layer
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        # Apply Log Softmax function to output layer
        x = self.output(x)
        x = F.log_softmax(x,dim=1)

        return x

def train(model, trainloader, validloader, criterion, optimizer, learning_rate=0.001, epochs=3)
