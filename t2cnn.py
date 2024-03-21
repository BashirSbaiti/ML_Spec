import gc
import itertools
import json
import random
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import pickle
from sklearn.metrics import f1_score
import argparse
from collections import OrderedDict
from models import resnet


class t2CNNDataset(Dataset):
    def __init__(self, DATALOC, LABELSLOC, POST_KEY, questionNum, randomSeed=None, isTrain=True,
                 split=.9,
                 dimersOnly=False, specificIndicies=[]):
        """
        Initiate an instance of a t2 Vectors Dataset, Inherits from Torch's dataset class so we can do easy data loading
        when training
        :param LABELSLOC: location of labels dataframe csv file (output from extract_labels.py) note this will also
        implicitly set what system indexes (within DATALOC) are being looked at through the keys of the dataframe
        :param DATALOC: location of presaved data to use. run generate_matrix.py to get this file
        :param POST_KEY: appended string to runname, indicating which postprocessing training data has undergone
        :param questionNum: question number to create labels from (1-6). Can be changed later with setNewY function
        :param randomSeed: for shuffling train/test data TRAIN AND TEST DATA MUST BE SHUFFLED WITH THE SAME SEED
        also note: you can disable shuffling by setting the seed equal to None
        :param isTrain: is it the training dataset?
        :param split: train/test split
        :param dimersOnly: include only dimers or not
        :param specificIndicies: list of specific indicies within LABELSLOC to look at, do not set to get all indicies
        within LABELSLOC. this param is only really useful for checking out a couple vectors for visualization
        """
        if specificIndicies is None:
            specificIndicies = []
        self.LABELSLOC = LABELSLOC
        self.POST_KEY = POST_KEY

        sysinfoDf = pd.read_csv(self.LABELSLOC, index_col=0)

        if len(specificIndicies) != 0:
            sysinfoDf = sysinfoDf.loc[[f"runJS{str(i).zfill(7)}" for i in specificIndicies], :]

        allDimers = list()  # list of all run inds which are dimers
        allMonomers = list()  # list of all run inds which are monomers
        for system in sysinfoDf.index:
            model = sysinfoDf.loc[system, "model"]
            if model.find("dimer") != -1:
                allDimers.append(int(system[5:12]))
            elif model.find("monomer") != -1:
                allMonomers.append(int(system[5:12]))
            else:
                raise ValueError(f"Unrecognized model {model}")

        self.indsSelected = allDimers if dimersOnly else sorted(allDimers + allMonomers)

        # shuffle systems
        if randomSeed is not None:
            rng = np.random.default_rng(seed=randomSeed)
            self.indsSelected = rng.permutation(self.indsSelected)

        # select appropriate subset of data for train/test
        if isTrain:
            self.indsSelected = self.indsSelected[0:int(split * len(self.indsSelected))]
        else:
            self.indsSelected = self.indsSelected[int(split * len(self.indsSelected)):len(self.indsSelected)]

        self.sysNamesSelected = [f"runJS{str(i).zfill(7)}{self.POST_KEY}" for i in self.indsSelected]
        self.NUM_SYSTEMS = len(self.sysNamesSelected)  # total num systems (after train/test split)

        # import precalculated t2 vectors
        xDataFname = f"xData_w1w3t3Matrix_{POST_KEY}.pkl"
        fullPath = f"{DATALOC}/{xDataFname}"

        if not os.path.exists(DATALOC):
            raise NotADirectoryError(f"DATALOC path {DATALOC} does not exist.")

        with open(fullPath, 'rb') as pklfile:
            xData = pickle.load(pklfile)
        # xData is dict of key system name, value matrix of shape w1, w3, t2

        # only retrieve systems to be part of the current (train/test) set
        xDataSelected = {}
        for sysName in self.sysNamesSelected:
            xDataSelected[sysName] = xData[sysName]

        self.w1_size, self.w3_size, self.t2_size = xDataSelected[next(iter(xDataSelected))].shape
        self.nSamples = self.NUM_SYSTEMS

        x = np.zeros([self.nSamples, 1, self.t2_size, self.w1_size, self.w3_size], dtype=np.float16)

        count = 0
        for sysName, sysMat in xDataSelected.items():
            # sysMat shape = (w1, w3, t2)
            for t2 in range(sysMat.shape[2]):
                x[count, 0, t2] = sysMat[:, :, t2]
            count += 1

        # FINAL x shape = (num training examples, 1, t2, w1, w3)  (1 is for 1 channel)
        # FINAL y shape = (num training examples, )

        self.nSamples = x.shape[0]
        self.x = torch.from_numpy(x).float()
        self.y = self.setNewY(questionNum)

        print(f"x shape = {self.x.shape}\ny shape = {self.y.shape}")

        if self.x.shape[0] != self.y.shape[0]:
            raise Exception(
                f"Shape of x data points {self.x.shape} is not equal to number of y data points {self.y.shape}")

    def setNewY(self, question):
        """
        this function allows us to update the dataset's labels for answering a new question without reprocessing
        all the x data
        """
        allSysNoPostKey = [el.replace(self.POST_KEY, "") for el in self.sysNamesSelected]

        labelsDf = pd.read_csv(self.LABELSLOC, index_col=0).loc[allSysNoPostKey, :]

        vibFreqDicts = []  # all vib freq dicts, listed in same order as self.allSysNames and by extension indsSelected
        maxNumVibrations = 0
        for systemName in labelsDf.index:
            vibFreqDict = labelsDf.loc[systemName, "vib_freq"]
            vibFreqDict = json.loads(vibFreqDict.replace('\'', '\"'))

            if len(vibFreqDict) > maxNumVibrations:
                maxNumVibrations = len(vibFreqDict)

            for key, val in vibFreqDict.items():
                vibFreqDict[key] = int(val)
            vibFreqDicts.append(vibFreqDict)

        if question == 1:
            allouts = np.array(labelsDf.loc[:, "model"] == "monomer", dtype=int)
        elif question == 2:
            allouts = []
            for vibFreqDict in vibFreqDicts:
                toAdd = 1 if 1300 in vibFreqDict.values() else 0
                allouts.append(toAdd)
        elif question == 3:
            allouts = []
            for vibFreqDict in vibFreqDicts:
                toAdd = 1 if 800 in vibFreqDict.values() else 0
                allouts.append(toAdd)
        elif question == 4:
            allouts = []
            for vibFreqDict in vibFreqDicts:
                toAdd = 1 if 200 in vibFreqDict.values() else 0
                allouts.append(toAdd)
        elif question == 5:
            allouts = []
            vfCategoryDict = {}
            i = 0
            for n in range(1, maxNumVibrations + 1):
                combs = itertools.combinations([1300, 200], n)
                for comb in combs:
                    vfCategoryDict[str(list(comb))] = i
                    i += 1

            for vibFreqDict in vibFreqDicts:
                toAdd = vfCategoryDict[str(list(vibFreqDict.values()))]
                allouts.append(toAdd)
        elif question == 6:
            allouts = []
            for system in labelsDf.index:
                J = labelsDf.loc[system, "J"]
                Jcat = -1
                if J < -650:
                    Jcat = 0
                elif J < -250:
                    Jcat = 1
                elif J < 250:
                    Jcat = 2
                elif J < 650:
                    Jcat = 3
                elif J <= 1000:
                    Jcat = 4

                if Jcat == -1:
                    raise Exception(f"Error: J coupling {J} is out of range.")
                else:
                    allouts.append(Jcat)
        else:
            raise Exception(f"Question {question} not implemented")

        y = np.array(allouts)

        self.y = torch.from_numpy(y).long()
        return self.y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.nSamples


class CNNchannel(nn.Module):
    def __init__(self, t2Size, numClasses):
        super(CNNchannel, self).__init__()
        # initialize first set of CONV -> RELU -> POOL layers
        self.conv1 = nn.Conv2d(in_channels=t2Size, out_channels=20,
                               kernel_size=(7, 7))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize second set of CONV -> RELU -> POOL layers
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=20,
                               kernel_size=(7, 7))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize first (and only) set of FC -> RELU layers
        self.fc1 = nn.Linear(in_features=21780, out_features=2000)  # TODO: remove hardcoded number
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=2000, out_features=numClasses)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # pass the input through our first set of CONV -> RELU ->
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # pass the output from the previous layer through the second
        # set of CONV -> RELU -> POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # flatten the output from the previous layer and pass it
        # through our only set of FC -> RELU layers
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output


class ConvBlock3d(nn.Module):
    def __init__(self, convShape: tuple, channelsIn: int, channelsOut: int, stride: int, padding: tuple):
        """
        :param convShape: shape of convolutions to do in (t2, w1, w3)
        :param channelsIn: number of input channels
        :param channelsOut: number of output channels
        """
        super().__init__()

        self.conv = nn.Conv3d(in_channels=channelsIn, out_channels=channelsOut, kernel_size=convShape, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        #self.mp = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        out = self.relu(x)
        #out = self.mp(x)
        return out


class ConvBlock2plus1D(nn.Module):
    def __init__(self, convShape2d: tuple, tempConv: int, channelsIn: int, channelsOut: int, stride: int, padding: tuple):
        """
        :param convShape: shape of convolutions to do in (w1, w3)
        :param tempConv: temporal conv length to do (depthwise in t2)
        :param channelsIn: number of input channels
        :param channelsOut: number of output channels
        """
        super().__init__()

        t = tempConv
        d = convShape2d[0]
        N = channelsIn
        N1 = channelsOut
        intermed = (t * (d ** 2) * N * N1) / (
                (d ** 2) * N + t * N1)  # used in https://arxiv.org/pdf/1711.11248.pdf (pg 4)

        self.spacialConv = nn.Conv3d(in_channels=channelsIn, out_channels=channelsOut,
                                     kernel_size=(1, convShape2d[0], convShape2d[1]), stride=(1, stride, stride),
                                     padding=(0, padding[1], padding[2]))
        self.temporalConv = nn.Conv3d(in_channels=channelsOut, out_channels=channelsOut,
                                      kernel_size=(tempConv, 1, 1), stride=(stride, 1, 1),
                                      padding=(padding[0], 0, 0))
        self.relu = nn.ReLU()
        # self.mp = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.spacialConv(x)
        x = self.relu(x)  # can go with or without, but paper says this is good due to extra nonlinearity
        x = self.temporalConv(x)
        out = self.relu(x)
        # out = self.mp(x)
        return out


class CNN3dLe(nn.Module):
    def __init__(self, inShape: tuple, convShape: tuple, numClasses: int, convBlocks: int, channels: tuple,
                 strides: tuple,
                 useGAP=False, decouplet2=False, t2padding=0):
        """
        :param inShape: shape of input data cube (t2, w1, w3)
        :param convShape: shape of convolutions to do in (t2, w1, w3)
        :param numClasses: number of final output classes
        :param convBlocks: number of (conv -> relu -> maxpool) blocks
        :param channels: number of output channels for each convolution, input data is assumed to be 1 channel
        :param useGAP: use global average pooling instead of flattening when connecting to fc layer.
        :param decouplet2: use 2+1D factorized convultions or not
        :param t2padding: number to zero pad on each size in t2 dimension
        """
        super().__init__()
        self.useGap = useGAP
        t2c, w1c, w3c = convShape
        currentShape = list(inShape)  # keep track of data shape throughout process
        self.channels = channels
        currentShape.insert(0, 1)  # <- represents 1 channel in initial data
        self.layers = OrderedDict()

        if len(self.channels) != convBlocks:
            raise Exception("Number of channels must equal number of conv blocks requested")
        if len(strides) != convBlocks:
            raise Exception("Number of strides provided must equal number of conv blocks requested")

        t2Collapsed = False
        # initialize a variable number of convBlocks (conv -> relu -> pool), while keeping track of shape
        for i in range(1, convBlocks + 1):
            nextShape = self.updateShape(currentShape, (t2c, w1c, w3c), self.channels[i - 1], strides=strides[i - 1],
                                         paddings=(t2padding, 0, 0))
            print(f"{i}: {nextShape}")
            # check if spacial conv will overflow input size or if t2 dimension has already been collapsed to one
            # (temporal conv makes no sense in this case!)
            if np.prod(nextShape[2:]) > 0 and not t2Collapsed:
                if nextShape[1] <= 0:
                    t2c = currentShape[1]
                    nextShape = self.updateShape(currentShape, (t2c, w1c, w3c), self.channels[i - 1],
                                                 strides=strides[i - 1])
                    print(
                        f"Warning: t2 conv overflow was avoided by changing temporal conv from {convShape[0]} to {t2c}")
                    t2Collapsed = True
                currentShape = nextShape
                inCh = self.channels[i - 2] if i - 2 >= 0 else 1

                if decouplet2:
                    spacialConv = (w1c, w3c)
                    temporalConv = t2c
                    self.layers[f"Conv Block {i}"] = ConvBlock2plus1D(spacialConv, temporalConv, inCh,
                                                                      self.channels[i - 1],
                                                                      strides[i - 1], padding=(t2padding, 0, 0))
                else:
                    self.layers[f"Conv Block {i}"] = ConvBlock3d((t2c, w1c, w3c), inCh, self.channels[i - 1],
                                                                 strides[i - 1], padding=(t2padding, 0, 0))
            else:
                print(
                    f"Warning: convolution {i} was skipped because conv size {convShape} > input shape {nextShape[1:]}.")

        if useGAP:
            self.layers["GAP"] = nn.AvgPool3d(kernel_size=tuple(currentShape[1:]), stride=1)
            self.layers["FC"] = nn.Linear(in_features=self.channels[-1], out_features=numClasses)
        else:
            self.layers["flatten"] = nn.Flatten()  # default params will flatten all but batch giving shape (batch, x)
            self.layers["FC1"] = nn.Linear(in_features=int(np.prod(currentShape)), out_features=1024)
            self.layers[f"ReLU_FC"] = nn.ReLU()
            self.layers[f"FC2"] = nn.Linear(in_features=1024, out_features=numClasses)

        self.fullNet = nn.Sequential(self.layers)

    def forward(self, x):
        output = self.fullNet(x)
        return output

    def updateShape(self, inShape: list, convShape: tuple, outChannels: int, paddings=(0, 0, 0), strides=1):
        if type(strides) is not tuple and type(strides) is not list:
            strides = [strides] * len(inShape)
        if type(paddings) is not tuple and type(paddings) is not list:
            paddings = [paddings] * len(inShape)
        if paddings == "same" and 1 in strides:
            raise Exception("Padding cannot be same if strides are not one.")

        outShape = []
        inShapeNoCh = inShape[1:].copy()  # pop off channel information
        for inDim, convDim, stride, pad in zip(inShapeNoCh, convShape, strides, paddings):
            if pad == "same":
                outDim = inDim
            else:
                nSteps = len(range(convDim - 1 - pad, inDim + pad, stride))
                outDim = nSteps
            outShape.append(outDim)
        outShape.insert(0, outChannels)
        return outShape


if __name__ == "__main__":

    ###########################
    ####   Pytorch Zone  ######
    ###########################

    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    args = parser.parse_args()
    RUNLABEL = args.label

    DATALOC = "/home/bs407/t2CNN/JS9042_11083Dataset/xData"
    datasetName = "JS9042_11083Dataset"

    labelsLoc = r"/home/bs407/t2CNN/JS9042_11083.csv"

    postKey = "trim"

    verbose = True

    useGAP = False
    decouplet2 = True
    useR2plus1_18 = False
    convShapes = [(30, 3, 3)]
    convBlocks = 6
    channels = (8, 16, 32, 64, 128, 256)
    strides = (1, 2, 2, 2, 2, 2)
    t2Pads = [30, 15, 5, 0]
    questions = [6]
    trainTestSplit = 0.8

    # TODO: work in train/dev/test
    randomSeed = random.randint(1, 100000)
    TrainDs = t2CNNDataset(DATALOC, labelsLoc, postKey, questions[0], randomSeed=randomSeed, isTrain=True,
                           split=trainTestSplit, dimersOnly=questions[0] == 6)
    TestDs = t2CNNDataset(DATALOC, labelsLoc, postKey, questions[0], randomSeed=randomSeed, isTrain=False,
                          split=trainTestSplit, dimersOnly=questions[0] == 6)

    for question in questions:
        if question != 6:
            TrainDs.setNewY(question)
            TestDs.setNewY(question)
        if question == 6 and len(questions) > 1:  # question 6 requires a dimer only subset of the data
            # before creating this new dataset, memory must be cleared or we will run out!
            # attempt to invoke garbage collection
            TrainDs = None
            del TrainDs
            TestDs = None
            del TestDs
            train_loader = None
            del train_loader
            test_loader = None
            del test_loader
            gc.collect()

            randomSeed = random.randint(1, 100000)
            TrainDs = t2CNNDataset(DATALOC, labelsLoc, postKey, question, randomSeed=randomSeed, isTrain=True,
                                   split=trainTestSplit, dimersOnly=question == 6)
            TestDs = t2CNNDataset(DATALOC, labelsLoc, postKey, question, randomSeed=randomSeed, isTrain=False,
                                  split=trainTestSplit, dimersOnly=question == 6)

        nExamples = len(TestDs) + len(TrainDs)

        t2Size = TrainDs.t2_size
        numClasses = -1
        if question == 6:
            numClasses = 5
        elif question == 5:
            numClasses = 8
        elif question in [1, 2, 3, 4]:
            numClasses = 2
        numEpochs = 10
        batchSize = 10  # one batch at least contains 10 systems
        lr = 0.001

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # dataloaders automatically shuffle and split data up into batches of requested size
        train_loader = DataLoader(dataset=TrainDs, batch_size=batchSize, shuffle=True)
        test_loader = DataLoader(dataset=TestDs, batch_size=1, shuffle=False)

        for convShape in convShapes:
            for t2pad in t2Pads:
                inShape = (TrainDs.t2_size, TrainDs.w1_size, TrainDs.w3_size)
                if useR2plus1_18:
                    model = resnet.r2plus1d_18(num_classes=numClasses)
                else:
                    model = CNN3dLe(inShape, convShape, numClasses, convBlocks, channels, strides,
                                    useGAP=useGAP, decouplet2=decouplet2, t2padding=t2pad).to(device)

                lossFn = nn.CrossEntropyLoss()  # THIS LOSS FUNCTION DOES BOTH ONEHOT ENCODING AND SOFTMAX FOR US
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                numSteps = len(train_loader)

                scaler = torch.cuda.amp.GradScaler()
                for epoch in range(numEpochs):
                    for i, (input, label) in enumerate(train_loader):

                        optimizer.zero_grad()

                        # loads in one batch
                        input = input.to(device)
                        label = label.to(device)

                        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                            outs = model(input)
                            # outs shape = batchsize, 2
                            loss = lossFn(outs, label)  # model applies softmax to outs at this point
                        with torch.no_grad():
                            _, preds = torch.max(outs, dim=1)
                            acc = torch.sum(preds == label) / label.size(dim=0) * 100.
                            f1 = f1_score(label, preds, average="macro")

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        if (verbose and i == 0) or (epoch == numEpochs - 1 and i == 0):
                            # TODO: report full epoch accuracy not batch accuracy
                            print(f"epoch {epoch + 1}/{numEpochs}, loss = {loss.item()}, acc = {acc}, f1 = {f1}",
                                  flush=True)

                # testing
                with torch.no_grad():
                    systemInds = TestDs.indsSelected
                    systemNames = [f"runJS{str(ind).zfill(7)}{postKey}" for ind in systemInds]
                    allPreds = list()
                    allLabels = list()
                    # TODO: add misclass df back in
                    totalCorrect = 0
                    totalSamples = 0
                    for index, (input, label) in enumerate(test_loader):
                        input = input.to(device)
                        label = label.to(device)
                        output = model(input)

                        # throw away max value, only max index is important
                        _, prediction = torch.max(output, 1)

                        allPreds.append(prediction.item())
                        allLabels.append(label.item())

                        totalSamples += 1
                        isCorrect = prediction == label
                        totalCorrect += 1 if isCorrect else 0

                    acc = 100.0 * totalCorrect / totalSamples
                    f1 = f1_score(allLabels, allPreds, average="macro")

                    print(f"answering question {question}", flush=True)
                    print(
                        f"CNN Specs: {model}, Batch size = {batchSize}, num classes = {numClasses}, lr = {lr}",
                        flush=True)

                    print(f"Validation Accuracy={acc}, F1 Score = {f1}\n\n", flush=True)

                    outDir = f"/home/bs407/t2CNN/{datasetName}/Confusion Histograms/Question {question}/CSVs"

                    if not os.path.exists(outDir):
                        os.makedirs(outDir)

                    # full of test data results, index = window index, col = system name
                    # element = int(np.nan) = -2147483648 if correctly classified and element = predicted if incorrectly classified
