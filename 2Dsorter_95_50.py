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
# Collecting the datasets
# The data is collected and formated to look like:
# inputs to NN: (K, 50, 50)
# TRUE outputs for NN: (K)
# K is the number of samples for training and testing (i.e. the total number of spectra)
#
# The output values correspond to the Coulombic couplings of the underlying system:
# 0 : [-1000, -650) cm^-1 
# 1 : [-650, -250) cm^-1
# 2 : [-250, 250] cm^-1
# 3 : (250, 650] cm^-1
# 4 : (650, 1000] cm^-1
###########################################################

    def __init__(self):
        for ii in range(175):
            filename1 = '/home/kp272/2DEV/Jon_NW_code/2022/LIBRARY/CSQC_MLcollab/DIMER_ML3/95/95_50/dim_95_50_ml3_' + str(ii+1) + '.mat'
            mat1 = scipy.io.loadmat(filename1)
            if ii == 0:
                xdata = mat1['Rw1w3Absorptive']
            else:
                xdata = np.concatenate((xdata, mat1['Rw1w3Absorptive']), axis=0)


        for ii in range(175):
            filename1 = '/home/kp272/2DEV/Jon_NW_code/2022/VALIDATION_SETS/CSQC_MLcollab/DIMER_ML2/95/95_50/dim_95_50_file_' + str(ii+1) + '.mat'
            mat1 = scipy.io.loadmat(filename1)
            xdata = np.concatenate((xdata, mat1['Rw1w3Absorptive']), axis=0)


        for ii in range(175):
            filename1 = '/home/kp272/2DEV/Jon_NW_code/2022/VALIDATION_SETS2/CSQC_MLcollab/DIMER_ML1/95/95_50/dim_95_50_file_' + str(ii+1) + '.mat'
            mat1 = scipy.io.loadmat(filename1)
            xdata = np.concatenate((xdata, mat1['Rw1w3Absorptive']), axis=0)


        for ii in range(175):
            filename1 = '/home/kp272/2DEV/Jon_NW_code/2022/DIMER_LIBRARY1/CSQC_MLcollab/MINI_LIB1/95/95_50/95_50_file_' + str(ii+1) + '.mat'
            mat1 = scipy.io.loadmat(filename1)
            xdata = np.concatenate((xdata, mat1['Rw1w3Absorptive']), axis=0)


        for ii in range(175):
            filename1 = '/home/kp272/2DEV/Jon_NW_code/2022/DIMER_VALIDATION_SETS1/CSQC_MLcollab/MINI_LIB1/95/95_50/95_50_file_' + str(ii+1) + '.mat'
            mat1 = scipy.io.loadmat(filename1)
            xdata = np.concatenate((xdata, mat1['Rw1w3Absorptive']), axis=0)



        with open('/home/kp272/2DEV/Jon_NW_code/2022/LIBRARY/CSQC_MLcollab/DIMER_ML3/output1.txt') as f:
            outdata = []
            for line in f: 
                outdata.append([int(x) for x in line.split()])

        ydataall = np.array(outdata)
        ydatans = ydataall[:,0]
        scaleit = np.ones(51)
        ydata = np.kron(ydatans, scaleit)


        with open('/home/kp272/2DEV/Jon_NW_code/2022/VALIDATION_SETS/CSQC_MLcollab/DIMER_ML2/output1.txt') as f:
            outdata = []
            for line in f: 
                outdata.append([int(x) for x in line.split()])

        ydataall = np.array(outdata)
        ydatans = ydataall[:,0]
        scaleit = np.ones(51)
        ydata1 = np.kron(ydatans, scaleit)
        ydata = np.concatenate((ydata, ydata1), axis=0)


        with open('/home/kp272/2DEV/Jon_NW_code/2022/VALIDATION_SETS2/CSQC_MLcollab/DIMER_ML1/output1.txt') as f:
            outdata = []
            for line in f: 
                outdata.append([int(x) for x in line.split()])

        ydataall = np.array(outdata)
        ydatans = ydataall[:,0]
        scaleit = np.ones(51)
        ydata2 = np.kron(ydatans, scaleit)
        ydata = np.concatenate((ydata, ydata2), axis=0)

  
        with open('/home/kp272/2DEV/Jon_NW_code/2022/DIMER_LIBRARY1/CSQC_MLcollab/MINI_LIB1/output1.txt') as f:
            outdata = []
            for line in f: 
                outdata.append([int(x) for x in line.split()])

        ydataall = np.array(outdata)
        ydatans = ydataall[:,0]
        scaleit = np.ones(51)
        ydata3 = np.kron(ydatans, scaleit)
        ydata = np.concatenate((ydata, ydata3), axis=0)


        with open('/home/kp272/2DEV/Jon_NW_code/2022/DIMER_VALIDATION_SETS1/CSQC_MLcollab/MINI_LIB1/output1.txt') as f:
            outdata = []
            for line in f: 
                outdata.append([int(x) for x in line.split()])

        ydataall = np.array(outdata)
        ydatans = ydataall[:,0]
        scaleit = np.ones(51)
        ydata4 = np.kron(ydatans, scaleit)
        ydata = np.concatenate((ydata, ydata4), axis=0)


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
# hyper parameters
###########################################################
input_size = 2500          # 50*50 -- the size of the input layer
hidden_size = 100          # size of the hidden layer
num_classes = 5            # size of the output layer
num_epochs = 100           # number of iterations
batch_size = 500           # number of inputs used in minibatches for training
learning_rate = 0.001      # step size
###########################################################


# device config
# use gpu if available, otherwise use cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# get datasets
full_dataset = TDDataset()

full_size = len(full_dataset)
test_size = int(51*175 + 51*175 + 51*175 + 51*175)
################### BS ####################
trainIndx = list(range(test_size))
random.shuffle(trainIndx)
testIndx = list(range(test_size, full_size))
random.shuffle(testIndx)
train_dataset = torch.utils.data.Subset(full_dataset, trainIndx)
test_dataset = torch.utils.data.Subset(full_dataset, testIndx)


# load the data for training and testing
# the training data is shuffled in minibatches used for training
# the testing data doesn't need to be shuffled
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# samples (spectra inputs) and labels (coupling ranges) are specified
examples = iter(train_loader)
samples, labels = examples.next()

# generate the model
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# flatten is used to convert the 50*50 to 25000
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
        images = flatten(images).to(device)
        labels = labels.to(device)        

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
#
allpredictions = []
allprobabilities = [[]] 
cccc = 0
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


        allpredictions = np.concatenate((allpredictions, predictions), axis=0)
        sm = torch.nn.Softmax(dim=1)
        probabilities = sm(outputs)
        newprobabilities = probabilities.numpy()
        if cccc == 0:
            allprobabilities = newprobabilities
        else:
            allprobabilities = np.concatenate((allprobabilities, newprobabilities), axis=0)
        cccc += 1

    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}')
    print(f'correct = {n_correct}')
    print(f'samples = {n_samples}')

    finprobabilities = allprobabilities.max(axis=1)
    finprobabilities *= 100





####### SAVING DATA FOR ANALYSIS
#
#with open('/home/kp272/2DEV/Jon_NW_code/2022/LIBRARY/CSQC_MLcollab/DIMER_ML3/output1.txt') as f:
#    outdata = []
#    for line in f: # read rest of lines
#        outdata.append([int(x) for x in line.split()])
#
#ydataall = np.array(outdata)
#ydatans = ydataall[:,0]
#scaleit = np.ones(51)
#newydata = np.kron(ydatans, scaleit)
#
#
#
#
#
#with open('/home/kp272/2DEV/Jon_NW_code/2022/VALIDATION_SETS/CSQC_MLcollab/DIMER_ML2/output1.txt') as f:
#    outdata = []
#    for line in f: # read rest of lines
#        outdata.append([int(x) for x in line.split()])
#
#ydataall = np.array(outdata)
#ydatans = ydataall[:,0]
#scaleit = np.ones(51)
#newydata1 = np.kron(ydatans, scaleit)
#newydata = np.concatenate((newydata, newydata1), axis=0)
#
#
#
#
#with open('/home/kp272/2DEV/Jon_NW_code/2022/VALIDATION_SETS2/CSQC_MLcollab/DIMER_ML1/output1.txt') as f:
#    outdata = []
#    for line in f: # read rest of lines
#        outdata.append([int(x) for x in line.split()])
#
#ydataall = np.array(outdata)
#ydatans = ydataall[:,0]
#scaleit = np.ones(51)
#newydata2 = np.kron(ydatans, scaleit)
#newydata = np.concatenate((newydata, newydata2), axis=0)
#
#
#
#
#with open('/home/kp272/2DEV/Jon_NW_code/2022/DIMER_LIBRARY1/CSQC_MLcollab/MINI_LIB1/output1.txt') as f:
#    outdata = []
#    for line in f: # read rest of lines
#        outdata.append([int(x) for x in line.split()])
#
#ydataall = np.array(outdata)
#ydatans = ydataall[:,0]
#scaleit = np.ones(51)
#newydata3 = np.kron(ydatans, scaleit)
#newydata = np.concatenate((newydata, newydata3), axis=0)
#
#
#
#
#with open('/home/kp272/2DEV/Jon_NW_code/2022/DIMER_VALIDATION_SETS1/CSQC_MLcollab/MINI_LIB1/output1.txt') as f:
#    outdata = []
#    for line in f: # read rest of lines
#        outdata.append([int(x) for x in line.split()])
#ydataall = np.array(outdata)
#ydatans = ydataall[:,0]
#scaleit = np.ones(51)
#newydata4 = np.kron(ydatans, scaleit)
#newydata = np.concatenate((newydata, newydata4), axis=0)
#
#
#
#from scipy.io import savemat
#savemat('predictions.mat', {'predictions': allpredictions, 'labels': newydata, 'percents': finprobabilities})
 
 
 
