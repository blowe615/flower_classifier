## Table of Contents
* [Installations](#installations)
* [Project Motivation](#project-motivation)
* [File Descriptions](#file-descriptions)
* [Using the Command Line](#using-the-command-line)
* [Results](#results)
* [Acknowledgements](#acknowledgements)

## Installation
All of the following packages were used for this project and can be found in the standard Anaconda distribution for Python 3.7:
* NumPy
* Matplotlib
* PIL
* Pytorch (torch and torchvision)

## Project Motivation

As part of Udacity's [Data Scientist Nanodegree](https://www.udacity.com/school-of-data-science) program, I created an image classifier based on a data set of 102 types of flowers.  The classifier consists of a deep convolutional neural net that was pretrained on ImageNet images and available through PyTorch's torchvision.models package.  For this project, I had to load and preprocess the image set, train the classifier on the image set, and then predict flower types for a given image.  These steps can be run either in a Jupyter notebook, or from the command line of a terminal.

## File Descriptions
Files 1-3 are required to run the Jupyter notebook.  Files 2-7 are required to run via command line.

1. `Image Classifier Project.ipynb`: Jupyter notebook that walks through all of the steps of preprocessing, training, and prediction.
2. `workspace_utils.py`: Contains functions provided by Udacity for keeping a workspace active while training on a GPU
3. `cat_to_name.json`: JSON file mapping the labels to flower names for the flower image set
4. `train.py`: Contains the general workflow for training an ImageNet model, testing its accuracy, and saving a model checkpoint.  Requires inputs from the command line.
5. `predict.py`: Contains the general workflow for predicting flower types for a given image based on a trained model.  Requires inputs from the command line.
6. `model_functions.py`: Creates a model class that contains the architecture of the classifier.  Also includes the functions to train, test, and predict.
7. `helper_functions.py`: Various functions to prepare for model training or prediction: argument parsers from the command line, loading and saving checkpoints, processing and unprocessing images.

## Using the Command Line
### Training a model
Train a new network on a data set with `train.py`
* Basic usage:

  `python train.py data_directory`
* Prints out training loss, validation loss, and validation accuracy as the network trains
* The `data_directory` must have folders within it labeled `train`, `valid`, and `test` and within each of those must be folders with index labels (1, 2, 3, etc) containing images
* Options:
  * Set directory to save checkpoints (default `checkpoint.pth`):

    `python train.py data_dir --save_dir save_directory`
  * Choose architecture (alexnet or vgg19_bn):

   `python train.py data_dir --arch "alexnet"`
  * Set hyperparameters (defaults: 0.001, 1000, 3):

   `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
  * Use GPU for training (default off):

    `python train.py data_dir --gpu`

### Predicting Flower Type
Predict flower name from an image with `predict.py` along with the probability of that name. You'll pass in a single image `/path/to/image` and return the flower name and class probability.

* Basic usage:

  `python predict.py /path/to/image checkpoint`
* Options:
  * Return top **K** most likely classes (default 5):

   `python predict.py input checkpoint --top_k 3`
  * Use a mapping of categories to real names (default `cat_to_name.json`):

   `python predict.py input checkpoint --category_names cat_to_name.json`
  * Use GPU for inference (default off):

    `python predict.py input checkpoint --gpu`

## Results
Running `train.py` will print out the epoch number, training loss, validation loss, and validation accuracy after every 10 batches of 32 images.

![](https://github.com/blowe615/flower_classifier/blob/master/train_results.png)

Running `predict.py` will output the image provided and the top **K** categories along with their probabilities.

![](https://github.com/blowe615/flower_classifier/blob/master/predict_example.png)

## Acknowledgements
Stack Overflow posts and the documentation for each of the python packages were extremely helpful in completing this project.
