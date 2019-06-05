# Import packages
import numpy as np
import matplotlib.pyplot as plt

import json
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from helper_functions import load_checkpoint, process_image, label_mapping
from model_functions import predict

# Load model checkpoint from path
print('Loading checkpoint...')
checkpoint = 'checkpoint_cli.pth'
model, optimizer, epochs = load_checkpoint(checkpoint)

# Process image to run through model
input_image = 'test_image.jpg'
image = process_image(input_image)

# Load dictionary mapping labels to category names
cat_to_name = 'cat_to_name.json'
label_dict = label_mapping(cat_to_name)

# Run the image through the model and obtain the topk predictions
topk = 5
# Identify the device on which the prediction will be performed
#GPU = arg_dict['gpu']
GPU = False
if GPU:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

probs, classes = predict(image, model, topk, device)

# Create list of labels for the topk classes
labels = []
for cls in classes:
    labels.append(label_dict[cls])

# Plot the cropped input image and the topk classes with probabilities
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(4,8))
# Undo transposition of dimensions required by PyTorch
image = image.transpose((1, 2, 0))
# Undo preprocessing
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
image = std * image + mean

# Image needs to be clipped between 0 and 1 or it looks like noise when displayed
image = np.clip(image, 0, 1)

# Turn off axes labels for cleaner image
#ax1.set_title(cat_to_name['1'])
ax1.tick_params(axis='both', length=0)
ax1.set_xticklabels('')
ax1.set_yticklabels('')

ax1.imshow(image)

ax2.barh(np.arange(len(probs)),probs)
ax2.invert_yaxis()
ax2.set_yticks(np.arange(len(probs)))
ax2.set_yticklabels(labels)
plt.show()
