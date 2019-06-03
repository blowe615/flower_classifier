import torch
from torch import nn
import torch.nn.functional as F

arch_input_dict = {'vgg19_bn': 25088,
                   'alexnet': 9216}
arch='vgg19_bn'
arch_input_dict[arch]
102

class DeepNetworkClassifier(nn.Module):
    def __init__(self, input_units, output_units, hidden_units=1000,p_drop=0.2):
        '''
        Builds a classifier for a pretrained deep neural network for the flower dataset

        Inputs
        ------
        arch: string, model name from torchvision.models, determines the number of inputs in the classifier
        hidden_units: int, the number of hidden units in the hidden layer
        '''
        super().__init__()
        # Create input layer with input units based on model architecture
        self.input = nn.Linear(input_units,hidden_units)
        # Create output layer with 102 outputs (for 102 flower classes)
        self.output = nn.Linear(hidden_units,output_units)
        # Define level of dropout
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, x):
        '''
        Performs a forward pass through the network and returns the log probabilities

        x: layer in model
        '''
        # Apply ReLU activation function and dropout to the input layer
        x = F.relu(self.input(x))
        x = self.dropout(x)
        # Apply Log Softmax function to output layer
        x = self.output(x)
        x = F.log_softmax(x,dim=1)

        return x

def train(model, trainloader, validloader, criterion, optimizer, epochs=3, device):
    '''
    Train the model

    Inputs
    -------
    model: torchvision model
    trainloader: PyTorch dataloader containing the training dataset
    validloader: PyTorch dataloader containing the validation dataset
    criterion: PyTorch criterion
    optimizer: PyTorch optimizer with learning rate
    epochs: int, number of passes of the training data through the network
    device: 'cuda' if GPU is specified, otherwise 'cpu'
    '''
    # Initialize some counters
    steps = 0
    running_loss = 0
    print_every = 10

    for epoch in range(epochs):
        for images, labels in trainloader:
            steps += 1
            # Send the images and labels to the GPU
            images, labels = images.to(device), labels.to(device)
            # Zero gradients for this step
            optimizer.zero_grad()
            # Perform a forward pass on the models
            log_ps = model(images)
            # Calculate loss
            loss = criterion(log_ps, labels)
            # Backpropagate error
            loss.backward()
            # Take next step
            optimizer.step()
            # Aggregate loss
            running_loss += loss.item()

            # Display results
            if steps % print_every == 0:
                # Set model to evaluate mode
                model.eval()
                # Initialize the validation loss and accuracy
                valid_loss = 0
                valid_acc = 0
                # Run validation dataset through the network
                with torch.no_grad():
                    for images, labels in validloader:
                        # Send the images and labels to the GPU
                        images_v, labels_v = images.to(device), labels.to(device)
                        # Perform forward pass with validation images
                        log_ps_valid = model.forward(images_v)
                        # Calculate validation loss and aggregate
                        loss = criterion(log_ps_valid, labels_v)
                        valid_loss += loss
                        # Calculate validation accuracy
                        # Calculate the probabilities from the log_probabilities
                        ps = torch.exp(log_ps_valid)
                        # Determine the top probability
                        top_p, top_class = ps.topk(1, dim=1)
                        # Compare top_class to label
                        valid_equality = top_class == labels_v.view(*top_class.shape)
                        # Calculate accuracy by aggregating the equalities
                        valid_acc += torch.mean(valid_equality.type(torch.FloatTensor)).item()

                # Print Results
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Training Loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation Accuracy: {valid_acc/len(validloader):.3f}")

                # Reset counter
                running_loss = 0
                # Return model to training mode to calculate grads
                model.train()

def test(model, testloader, device):
    '''
    Test the model on the test dataset

    Inputs
    -------
    model: torchvision model
    testloader: PyTorch dataloader containing the testing dataset
    device: 'cuda' if GPU is specified, otherwise 'cpu'
    '''
    # Set model to evaluate mode
    model.eval()
    # Initialize the testing accuracy
    test_acc = 0

    # Run test dataset through the network
    with torch.no_grad():
        for images, labels in testloader:
            # Send the images and labels to the GPU
            images_t, labels_t = images.to(device), labels.to(device)
            # Perform forward pass with validation images
            log_ps_test = model.forward(images_t)
            # Calculate test accuracy
            # Calculate the probabilities from the log_probabilities
            ps_test = torch.exp(log_ps_test)
            # Determine the top probability
            top_p, top_class = ps_test.topk(1, dim=1)
            # Compare top_class to label
            test_equality = top_class == labels_t.view(*top_class.shape)
            # Calculate accuracy by aggregating the equalities
            test_acc += torch.mean(test_equality.type(torch.FloatTensor)).item()

    # Print Results
    print("Test Accuracy: {:.3f}".format(test_acc/len(testloader)))

    # Return model to training mode to calculate grads
    model.train();
