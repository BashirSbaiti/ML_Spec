###################################################################
#   NN code to characterize 2D spectra outputs from OREOS         #
#   Version 4-19-2022                                             #
###################################################################
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import scipy.io
import random

###########################################################
class TDDataset(Dataset):
###########################################################
    """
    Collecting the datasets
        The data is collected in the format:
            inputs: (T, n, n) where:
                T = the number of spectra in total 
                n  = number of ticks along excitation and emission axes for all spectra 
            outputs: (T, k)
                k = the size of the output layer
    """
###########################################################

    def __init__(self):
        # load file, load data set
        mat1 = scipy.io.loadmat('NN_example.mat')
        xdata = mat1['xdata']
        ydata = mat1['ydata'][0]
        print(ydata.shape)
        """
        This example uses NN_example.mat, which contains a subset of data described in the manuscript. The subset is monomers with one vibration described by Table 1 in the manuscript. Each system has 10 waiting times included (10 spectra). Datasets 1-3 have 10 systems each (30 in total); 15 have a high frequency mode and 15 do not. The ydata is the label: 0 = absent and 1 = present. The total number of spectra is 300 (3 Datasets * 10 systems * 10 waiting times). The spectra have been trimmed to 90% and resized to have excitation and emission axes of size 50 by 50 using 2Dspec_adjust.py.
        """

#       convert to data 'torch tensors'
        self.x = torch.from_numpy(xdata)
        self.y = torch.from_numpy(ydata)

#       convert to float32
        self.x = self.x.float()
        self.y = self.y.long()

#       total number of samples for both training and testing
        self.n_samples = self.x.shape[0]

    def __getitem__(self,index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

###########################################################
###########################################################
class NeuralNet(nn.Module):
###########################################################
# The feedforward NN
# The hidden layer(s) and activation functions are specified
###########################################################
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
###########################################################

###########################################################
# hyperparameters
###########################################################
input_size = 2500          # 50*50 -- the size of the input layer
hidden_size = 500          # size of the hidden layer
num_classes = 2            # size of the output layer
num_epochs = 100           # number of iterations
batch_size = 50            # number of inputs used in minibatches for training
learning_rate = 0.001      # step size
###########################################################

# device config
# use gpu if available, otherwise use cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# get datasets for training and testing
full_dataset = TDDataset()

full_size = len(full_dataset)
train_size = 200 

train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))             # training data set 
test_dataset = torch.utils.data.Subset(full_dataset, range(train_size, full_size))   # testing data set 

"""
For this example, the training dataset was the first 200 spectra defined in TDDataset and the testing set was the remaining 100 spectra. 
"""

# load the data for training and testing
# the training data is shuffled in minibatches
# the testing data doesn't need to be shuffled
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# samples (spectra inputs) and labels are specified
examples = iter(train_loader)
samples, labels = examples.next()

# generate the model
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# flatten is used to convert the 50*50 pixels into 2500 neurons
flatten = nn.Flatten()

# loss and optimizer are specified
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

###########################################################
###########################################################
# training loop
###########################################################
###########################################################
n_total_steps = len(train_loader)

# step through iterations
for epoch in range(num_epochs):
    # step through minibatches
    for i, (images, labels) in enumerate(train_loader):
        images = flatten(images).to(device)                # convert from 50*50 to 2500 and feed to device
        labels = labels.to(device)                         # feed labels to device (cpu or gpu)

        # forward steps through the neurons
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backwards steps to optimize the weights and biases
        # gradients need to be zeroed out
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss function (cost) periodically
        if (i+1) % 5 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

###########################################################
###########################################################
# test data results
###########################################################
###########################################################
#
# gradients turned off
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = flatten(images).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item() 

    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}')
    print(f'correct = {n_correct}')
    print(f'samples = {n_samples}')





