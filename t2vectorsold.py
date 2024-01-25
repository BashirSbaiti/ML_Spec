import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time

ON_CLUSTER = True

# these are 114x114 spectra, not zoomed in for now
# TODO: check out zoomed data
CLUS_DS1 = r"/home/kp272/2DEV/Jon_NW_code/2022/LIBRARY/CSQC_MLcollab/BIG_LIB3/114"
CLUS_DS2 = r"/home/kp272/2DEV/Jon_NW_code/2022/VALIDATION_SETS/CSQC_MLcollab/BIG_LIB2/114"
CLUS_DS3 = r"/home/kp272/2DEV/Jon_NW_code/2022/VALIDATION_SETS2/CSQC_MLcollab/BIG_LIB1/114"
CLUS_DS4 = r"/home/kp272/2DEV/Jon_NW_code/2022/DIMER_LIBRARY1/CSQC_MLcollab/BIG_LIB1/114"  # ONLY DIMERS
CLUS_DS5 = r"/home/kp272/2DEV/Jon_NW_code/2022/DIMER_VALIDATION_SETS1/CSQC_MLcollab/BIG_LIB1/114"  # ONLY DIMERS

# I = indiv output (questions 1-4), C = combo output (question 5), D = dimer output (question 6, only for dimers)
CLUS_DS1_LABELI = r"/home/kp272/2DEV/Jon_NW_code/2022/LIBRARY/CSQC_MLcollab/BIG_LIB3/indiv_output.txt"
CLUS_DS1_LABELC = r"/home/kp272/2DEV/Jon_NW_code/2022/LIBRARY/CSQC_MLcollab/BIG_LIB3/combo_output.txt"
CLUS_DS2_LABELI = r"/home/kp272/2DEV/Jon_NW_code/2022/VALIDATION_SETS/CSQC_MLcollab/BIG_LIB2/indiv_output.txt"
CLUS_DS2_LABELC = r"/home/kp272/2DEV/Jon_NW_code/2022/VALIDATION_SETS/CSQC_MLcollab/BIG_LIB2/combo_output.txt"
CLUS_DS3_LABELI = r"/home/kp272/2DEV/Jon_NW_code/2022/VALIDATION_SETS2/CSQC_MLcollab/BIG_LIB1/indiv_output.txt"
CLUS_DS3_LABELC = r"/home/kp272/2DEV/Jon_NW_code/2022/VALIDATION_SETS2/CSQC_MLcollab/BIG_LIB1/combo_output.txt"
CLUS_DS4_LABELD = r"/home/kp272/2DEV/Jon_NW_code/2022/DIMER_LIBRARY1/CSQC_MLcollab/BIG_LIB1/output1.txt"
CLUS_DS5_LABELD = r"/home/kp272/2DEV/Jon_NW_code/2022/DIMER_VALIDATION_SETS1/CSQC_MLcollab/BIG_LIB1/output1.txt"

FLUFF_DS1 = "114_bl3_"
FLUFF_DS2 = "114_file_"
FLUFF_DS3 = "114_file_"
FLUFF_DS4 = "114_file_"
FLUFF_DS5 = "114_file_"

# dimer only versions of DS 1-3 for question 6
CLUS_DS1_DIMERS = r"/home/kp272/2DEV/Jon_NW_code/2022/LIBRARY/CSQC_MLcollab/DIMER_BL3"
CLUS_DS2_DIMERS = r"/home/kp272/2DEV/Jon_NW_code/2022/VALIDATION_SETS/CSQC_MLcollab/DIMER_BL2"
CLUS_DS3_DIMERS = r"/home/kp272/2DEV/Jon_NW_code/2022/VALIDATION_SETS2/CSQC_MLcollab/DIMER_BL1"

CLUS_DS1_LABELD = r"/home/kp272/2DEV/Jon_NW_code/2022/LIBRARY/CSQC_MLcollab/DIMER_BL3/output1.txt"
CLUS_DS2_LABELD = r"/home/kp272/2DEV/Jon_NW_code/2022/VALIDATION_SETS/CSQC_MLcollab/DIMER_BL2/output1.txt"
CLUS_DS3_LABELD = r"/home/kp272/2DEV/Jon_NW_code/2022/VALIDATION_SETS2/CSQC_MLcollab/DIMER_BL1/output1.txt"

FLUFF_DS1_DIMERS = "dim_114_bl3_"
FLUFF_DS2_DIMERS = "dim_114_file_"
FLUFF_DS3_DIMERS = "dim_114_file_"


#### Kelsey's setup ####
# Questions 1-5: used DS1-2 for training and DS3 for testing
# Questions 6 (dimers): use DS4-5 and use just dimers from 1-3 (these are in separate directories)

def loadSystemData(i, dir, fluff):
    # loads ith system from the specified directory
    # fluff is a string phrase that is in the filename that's everything except for index integer
    # returns: (1, 51, 114, 114) array that describes the system

    fullpath = dir + f"/{fluff}{i}"
    mat = scipy.io.loadmat(fullpath)
    idata = mat["Rw1w3Absorptive"]

    return idata

    # shape of xdata = (system index, t2, w1, w3)
    #                       = (246, 51, 114, 114)


class t2Dataset(Dataset):
    def __init__(self, questionNum, winSize, step, shuffledInd, isTrain=True, split=.9, multiprocess=True,
                 dimers=False):
        """
        Initiate an instance of a t2 Vectors Dataset, Inherits from Torch's dataset class so we can do easy data loading
        when training
        :param questionNum: question number to create labels from (1-6). Can be changed later with getNewY function
        :param winSize: window size to make t2 vectors
        :param step: step size between windows
        :param shuffledInd: 1 INDEXED list of shuffled indexes for each system number (with a contin numbering scheme)
        :param isTrain: is the training dataset
        :param split: train/test split
        :param multiprocess: use multiprocessing or not
        :param dimers: include only dimers or not
        """
        if dimers:
            xdirs = [CLUS_DS1_DIMERS, CLUS_DS2_DIMERS, CLUS_DS3_DIMERS, CLUS_DS4, CLUS_DS5]
            self.yfiles = [CLUS_DS1_LABELD, CLUS_DS2_LABELD, CLUS_DS3_LABELD, CLUS_DS4_LABELD,
                           CLUS_DS5_LABELD]  # this has to be self to update y method can access it
            fluffs = [FLUFF_DS1_DIMERS, FLUFF_DS2_DIMERS, FLUFF_DS3_DIMERS, FLUFF_DS4, FLUFF_DS5]
            if questionNum != 6:
                raise Exception("The dataset was constructed with only dimer data, so model can only answer question 6,"
                                f" not question {questionNum} as requested.")
        else:
            xdirs = [CLUS_DS1, CLUS_DS2, CLUS_DS3]
            self.yfiles = [[CLUS_DS1_LABELI, CLUS_DS2_LABELI, CLUS_DS3_LABELI],
                           [CLUS_DS1_LABELC, CLUS_DS2_LABELC, CLUS_DS3_LABELC]]
            fluffs = [FLUFF_DS1, FLUFF_DS2, FLUFF_DS3]

        dsSizeList = list()
        for i, dir in enumerate(xdirs):
            onlyDataFiles = [1 if x.find(".mat") != -1 and x.find(fluffs[i]) != -1 else 0 for x in os.listdir(dir)]
            dsSizeList.append(sum(onlyDataFiles))
        dsSizeList = np.array(dsSizeList)

        self.NUM_SYSTEMS = dsSizeList.sum()

        if len(shuffledInd) != self.NUM_SYSTEMS:
            raise ValueError(f"Indices given are length {len(shuffledInd)}, but the total number of systems is "
                             f"{self.NUM_SYSTEMS}. These values should be equal.")

        # define the system numbers we're dealing with
        if isTrain:
            self.inds = shuffledInd[0:int(split * len(shuffledInd))]
        else:
            self.inds = shuffledInd[int(split * len(shuffledInd)):len(shuffledInd)]

        # Note: inds will use a continuous numbering scheme: indices 1-247 will bs DS1, 248-... will be DS2, etc.
        lastIndices = list()
        lastIndices.append(dsSizeList[0])  # indices start at 1 so the size is the last valid index
        for i, dsSize in enumerate(dsSizeList[1:], start=1):
            lastIndices.append(dsSizeList[i] + lastIndices[i - 1])

        if not multiprocess:
            start = time.perf_counter()
            self.xdata = np.array([])

            for i in self.inds:
                # check what system number we are dealing with
                dsNum = 1
                for num, lastIndex in enumerate(lastIndices, start=1):
                    if i > lastIndex:
                        dsNum = num
                    else:
                        break

                dsIndex = dsNum - 1  # the datasets start numbering at 1 but everything else starts at 0
                sysNumberinDs = i if dsIndex == 0 else i - lastIndices[i - 1]  # this is one indexed too
                idata = loadSystemData(sysNumberinDs, xdirs[dsIndex], fluffs[dsIndex]).reshape(1, 51, 114, 114)
                self.xdata = np.concatenate([self.xdata, idata], axis=0) if self.xdata.size > 1 else idata
            stop = time.perf_counter()
        else:
            # my attempt at multiprocessing
            from pathos.multiprocessing import ProcessingPool
            start = time.perf_counter()
            pool = ProcessingPool()
            with pool as executor:
                ivals = np.concatenate([np.arange(1, maxInd + 1, 1) for maxInd in dsSizeList])
                allDirs = np.array([[xdirs[i]] * dsSizeList[i] for i in range(len(xdirs))]).flatten()
                allFluffs = np.array([[fluffs[i]] * dsSizeList[i] for i in range(len(xdirs))]).flatten()
                results = executor.map(loadSystemData, ivals, allDirs, allFluffs)
                # results ends up stacked in such a way that it is equivalent to concatenating the systems on axis 0
                # so the shape ends up being NUM_SYSTEMS, 51, 114, 114 without any np reshape or concat
                results = np.array(results)
                self.xdata = results[self.inds - 1][:]  # apply shuffling
                # subtract 1 because self.inds is one indexed

        # chopping up the data
        windowSize = winSize  # windows are square
        step = step

        self.NUM_WINDOWS = 0

        # dictionary containing key = system number, value = array of vectors (vector shape 51, array shape (numslices, 51)
        sysDict = dict()

        # optimization note: this loop is actually quite fast
        for systemNum in range(self.xdata.shape[0]):
            matSys = self.xdata[systemNum]
            # shape = t2, w1, w3
            for row in range(0, matSys.shape[1] - windowSize + 1, step):
                rstart = row
                rstop = row + windowSize
                for col in range(0, matSys.shape[2] - windowSize + 1, step):
                    self.NUM_WINDOWS += 1
                    cstart = col
                    cstop = col + windowSize
                    window = matSys[:, rstart:rstop, cstart:cstop]
                    norm = np.linalg.norm(window, ord="fro",
                                          axis=(1, 2))  # compute frobenius norm with matrix = axis 1,2
                    # norm is now a 1d vector with size 51
                    if systemNum not in sysDict:
                        sysDict[systemNum] = norm.reshape(1, 51)
                    else:
                        sysDict[systemNum] = np.concatenate((sysDict[systemNum], norm.reshape(1, 51)), axis=0)

        self.NUM_WINDOWS = int(self.NUM_WINDOWS / self.xdata.shape[0])

        for i, key in enumerate(sysDict):
            if i == 0:
                x = sysDict[key]
            else:
                x = np.concatenate((x, sysDict[key]), axis=0)

        # FINAL x shape = (num training examples, t2 axis = 51)
        #                   windows per sys (self.NUM_WINDOWS) * num sys, 51
        #     (for 3x3)     1,444 * 246 = 355224, 51

        # FINAL y shape = (num training examples, )

        self.nSamples = x.shape[0]
        self.x = torch.from_numpy(x).float()
        self.y = self.setNewY(questionNum)

        if self.x.shape[0] != self.y.shape[0]:
            raise Exception(
                f"Number of x data points {self.x.shape[0]} is not equal to number of y data points {self.y.shape[0]}")

    def setNewY(self, question):
        # TODO: implement label retreval for questions 5 and 6
        """
        this function allows us to update the dataset's labels for answering a new question without reprocessing
        all the x data
        """
        if question == 6:
            allouts = np.array([])
            for yfile in self.yfiles:
                with open(yfile) as f:
                    outi = np.array([float(line[0]) for line in f], dtype=np.float16)
                    allouts = np.concatenate([allouts, outi], axis=0) if allouts.size >= 1 else outi

            allouts = allouts[self.inds - 1][:]  # SUFFLING: select out only specified (random) indexes from suffledInd
            # 1 was subtracted because inds is one indexed, we need 0 indexed for numpy
            scaleArr = np.ones(
                self.NUM_WINDOWS)  # scale knowing that each system gives us NUM_WINDOWS new training examples

            # kron = for each element of a, multiply it by every element of b individually
            y = np.kron(allouts, scaleArr)
            # basically makes NUM_WINDOWS copies of each number
            # New shape is now (NSystems*NWindows,)

            self.y = torch.from_numpy(y).long()
            return self.y
        elif question == 5:
            # this requires using combo output
            # this also requires changing net output to softmax to output [0-7] rather than [0, 1]
            # TODO: implement
            pass
        else:
            # for questions 1-4, so binary outputs for monomer systems
            allouts = np.array([])
            for yfile in self.yfiles[0]:
                with open(yfile) as f:
                    outi = np.array([line.split() for line in f], dtype=np.float16)
                    allouts = np.concatenate([allouts, outi], axis=0) if allouts.size > 1 else outi

            allouts = allouts[self.inds - 1][:]  # SUFFLING: select out only specified (random) indexes from suffledInd
            # 1 was subtracted because inds is one indexed, we need 0 indexed for numpy

            # SPLIT OUT LABELS FOR QUESTION questionNum (numbering starts at 1)
            allouts = allouts[:, question - 1]

            scaleArr = np.ones(
                self.NUM_WINDOWS)  # scale knowing that each system gives us NUM_WINDOWS new training examples

            # kron = for each element of a, multiply it by every element of b individually
            y = np.kron(allouts, scaleArr)
            # basically makes NUM_WINDOWS copies of each number
            # New shape is now (NSystems*NWindows,)

            self.y = torch.from_numpy(y).long()
            return self.y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.nSamples


class NeuralNet(nn.Module):

    def __init__(self, inputSize, hiddenSize, numClasses):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(inputSize,
                            hiddenSize)  # linear transform in hidden layer (between orig data and input into hidden) 1
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hiddenSize, numClasses)  # linear transform in output layer (between hidden 1 and out)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


if __name__ == "__main__":

    ###########################
    ####   Pytorch Zone  ######
    ###########################

    verbose = True
    questions = [6]
    winSizes = [60]
    stepFracs = [1]
    useDimers = True
    # If you use dimers MUST be asking question 6
    trainTestSplit = 0.8

    # for regular case
    SYSTEMS = 738 if not useDimers else 875
    indx = np.random.permutation(np.arange(1, SYSTEMS + 1, 1))  # needs to be one indexed because the first system
    # in the files is called 1

    totalRuns = len(questions) * len(winSizes) * len(stepFracs)
    runNum = 0

    for winSize in winSizes:
        for stepFrac in stepFracs:
            if winSize == 3 and stepFrac == .25:
                continue
            step = int(stepFrac * winSize)

            # for some reason this multiprocessing gets pretty slow with the second call to build the test set
            # but on the cluster it is fast so not sure what's happening there
            # TODO: work in train/dev/test
            TrainDs = t2Dataset(questions[0], winSize, step, indx, isTrain=True, split=trainTestSplit,
                                multiprocess=True,
                                dimers=useDimers)
            TestDs = t2Dataset(questions[0], winSize, step, indx, isTrain=False, split=trainTestSplit,
                               multiprocess=True,
                               dimers=useDimers)

            for question in questions:

                runNum += 1

                TrainDs.setNewY(question)
                TestDs.setNewY(question)
                nExamples = len(TestDs) + len(TrainDs)

                inSize = 51  # t2 timeseries
                hiddenSize = 32
                numClasses = -1
                if question == 6:
                    numClasses = 5
                elif question == 5:
                    numClasses = 8
                elif question in [1, 2, 3, 4]:
                    numClasses = 2
                numEpochs = 100
                batchSize = TrainDs.NUM_WINDOWS * 10  # one batch at least contains 10 systems
                lr = 0.001

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # dataloaders automatically shuffle and split data up into batches of requested size
                train_loader = DataLoader(dataset=TrainDs, batch_size=batchSize, shuffle=True)
                test_loader = DataLoader(dataset=TestDs, batch_size=1, shuffle=False)

                model = NeuralNet(inSize, hiddenSize, numClasses).to(device)

                lossFn = nn.CrossEntropyLoss()  # THIS LOSS FUNCTION DOES BOTH ONEHOT ENCODING AND SOFTMAX FOR US
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                numSteps = len(train_loader)

                for epoch in range(numEpochs):

                    for i, (inputs, labels) in enumerate(train_loader):
                        # loads in one batch
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outs = model(inputs)
                        # outs shape = batchsize, 2
                        loss = lossFn(outs, labels)  # model applies softmax to outs at this point
                        with torch.no_grad():
                            _, preds = torch.max(outs, dim=1)
                            acc = torch.sum(preds == labels) / labels.size(dim=0) * 100.

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        if verbose and i == 0:
                            print(f"epoch {epoch}/{numEpochs}, loss = {loss.item()}, acc = {acc}")

                # testing
                with torch.no_grad():
                    systemInds = TestDs.inds
                    windowsPerSys = TestDs.NUM_WINDOWS
                    misclassified = {}
                    totalCorrect = 0
                    totalSamples = 0
                    currentSys = 0
                    currentWindowInSys = 0
                    for index, (inputs, labels) in enumerate(test_loader):
                        currentWindowInSys += 1
                        if currentWindowInSys == windowsPerSys + 1:
                            currentSys += 1
                            currentWindowInSys = 1
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = model(inputs)

                        # throw away max value, only max index is important
                        _, predictions = torch.max(outputs, 1)
                        totalSamples += labels.shape[0]
                        icorrect = (predictions == labels).sum().item()
                        totalCorrect += icorrect

                        if icorrect == 0:
                            if currentSys not in misclassified.keys():
                                misclassified[currentSys] = list()
                            misclassified[currentSys].append(currentWindowInSys)

                    acc = 100.0 * totalCorrect / totalSamples

                    dir = f"Confusion Histograms/Question {question}/WinSize{winSize}"
                    if not os.path.exists(dir):
                        os.makedirs(dir)

                    misclassifiedAllSys = list()
                    for system in misclassified.keys():
                        misclassifiedAllSys += misclassified[system]  # collect all win indexes miscalssified
                        plt.figure()
                        plt.hist(misclassified[system])
                        plt.xlabel("Window Index")
                        plt.ylabel("Misclassifications")
                        plt.title(f"Confusion by position (system {system})")
                        #plt.savefig(f"{dir}/{winSize}System{system}.png")  # plot loc of misclassified for 1 sys
                        plt.close()

                    plt.figure()
                    plt.hist([misclassifiedAllSys])
                    plt.xlabel("Window Index")
                    plt.ylabel("Misclassifications")
                    plt.title("Confusion by position (overall)")
                    #plt.savefig(f"{dir}/{winSize}overallbyPos.png")
                    plt.close()

                    plt.figure()
                    plt.bar(misclassified.keys(), [len(x) for x in misclassified.values()])  # number of miscalssifcations per system
                    plt.xlabel("System")
                    plt.ylabel("Misclassifications")
                    plt.title("Confusion by system")
                    #plt.savefig(f"{dir}/{winSize}overallbySys.png")
                    plt.close()


                    print(f'answering question {question}')
                    print(f"NN Specs: 1 Hidden layer size {hiddenSize}, Batch size = {batchSize}, num classes = {numClasses}, lr = {lr}")

                    print(f"WinSize={winSize}, Step={step}, Step Fraction={stepFrac}, Validation Accuracy={acc}\n\n")
