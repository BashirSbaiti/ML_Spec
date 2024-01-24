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
parser = argparse.ArgumentParser()
parser.add_argument("label")
args = parser.parse_args()
RUNLABEL = args.label


newData = False

if newData:
    DATALOC = r"/home/jds199/ML_Database/Database/Simulated/postprocessed"  # new data 8002-9042
else:
    DATALOC = r"/home/jds199/NO_BACKUP/Database_nobackup/Simulated/postprocessed"  # 1041


def loadSystemData(name, DATALOC):
    # loads system with specific run name from the directory DATALOC
    # returns: (1, w1, w3, t2) array that describes the system

    fullpath = f"{DATALOC}/{name}.pkl"

    with open(fullpath, "rb") as pklfile:
        fullEntry = pickle.load(pklfile)
        idata = fullEntry['R']["Rw1w3Absorptive"]

    return idata

    # shape of xdata = (num systems, w1, w3, t2)


class t2Dataset(Dataset):
    def __init__(self, LABELSLOC, POST_KEY, questionNum, winSize, step, randomSeed=None, isTrain=True, split=.9,
                 dimersOnly=False, specificIndicies=[], normalizeVectors=False, lowRangeFilter=None,
                 highRangeFilter=None, RANGESLOC=None, saveUsedWindows=True):
        """
        Initiate an instance of a t2 Vectors Dataset, Inherits from Torch's dataset class so we can do easy data loading
        when training
        :param LABELSLOC: location of labels dataframe csv file (output from extract_labels.py) note this will also
        implicitly set what system indexes (within DATALOC) are being looked at through the keys of the dataframe
        :param POST_KEY: appended string to runname, indicating which postprocessing training data has undergone
        :param questionNum: question number to create labels from (1-6). Can be changed later with setNewY function
        :param winSize: window size to make t2 vectors
        :param step: step size between windows
        :param randomSeed: for shuffling train/test data TRAIN AND TEST DATA MUST BE SHUFFLED WITH THE SAME SEED
        also note: you can disable shuffling by setting the seed equal to None
        :param isTrain: is it the training dataset?
        :param split: train/test split
        :param dimersOnly: include only dimers or not
        :param specificIndicies: list of specific indicies within LABELSLOC to look at, do not set to get all indicies
        within LABELSLOC. this param is only really useful for checking out a couple vectors for visualization
        :param normalizeVectors: weather to normalize vectors or not. normalization is carried out by subtracting by the
        minimum value and dividing by the range (max-min value)
        :param lowRangeFilter: for each system, lowRangeFilter percent of the lowest range windows will be dropped. expressed as fraction of 1.0. Set to None to remove limit.
        :param highRangeFilter: for each system, highRangeFilter percent of the highest range windows will be dropped. expressed as fraction of 1.0. Set to None to remove limit.
        :param RANGESLOC: location of ranges dataframe csv file (output from quantify_windows.py). Set to None when range values will not be needed.
        :param saveUsedWindows: weather to save csv file containing information about what windows were used. only has an effect
        in the event that lowRangeFilter or highRangeFilter are defined.
        """
        self.LABELSLOC = LABELSLOC
        self.RANGESLOC = RANGESLOC
        self.POST_KEY = POST_KEY

        if RANGESLOC is None and normalizeVectors:
            raise Exception("No file path was provided to ranges csv, so vectors cannot be normalized. Please run "
                            "quantify_windows.py to generate the ranges csv.")
        elif RANGESLOC is None and (lowRangeFilter is not None or highRangeFilter is not None):
            raise Exception("No file path was provided to ranges csv, so windows cant be cut bases on ranges. Please "
                            "run quantify_windows.py to generate the ranges csv.")


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

        self.NUM_SYSTEMS = len(self.indsSelected)  # total num systems (before train/test split)

        # select appropriate subset of data for train/test
        if isTrain:
            self.indsSelected = self.indsSelected[0:int(split * len(self.indsSelected))]
        else:
            self.indsSelected = self.indsSelected[int(split * len(self.indsSelected)):len(self.indsSelected)]

        self.allSysNames = [f"runJS{str(i).zfill(7)}{self.POST_KEY}" for i in self.indsSelected]

        from pathos.multiprocessing import ProcessingPool
        start = time.perf_counter()
        pool = ProcessingPool(nodes=20)  # TODO: autodetect available cores?
        with pool as executor:
            names = [f"runJS{str(i).zfill(7)}{POST_KEY}" for i in self.indsSelected]
            print(f"beginning multiprocess at {start} sec", flush=True)
            results = executor.map(loadSystemData, names, [DATALOC] * len(names))
            # results ends up stacked in such a way that it is equivalent to concatenating the systems on axis 0
            # so the shape ends up being NUM_SYSTEMS, w1, w3 without any np reshape or concat
            results = np.array(results)
            self.xdata = results
            print(f"ending multiprocess after {time.perf_counter() - start} sec", flush=True)

        # chopping up the data
        windowSize = winSize  # windows are square
        step = step

        self.NUM_WINDOWS = int(self.xdata.shape[1] / windowSize) ** 2

        # dictionary containing key = system number, value = array of vectors (vector shape t2, array shape (numslices, t2)
        sysDict = dict()

        if self.RANGESLOC is not None:
            rangesDf = pd.read_csv(self.RANGESLOC, index_col=0).loc[:, self.allSysNames]

        # cut out windows according to low and high range filters
        if lowRangeFilter is not None or highRangeFilter is not None:
            allFilterSers = []
            for sysName in rangesDf.columns:
                rangesSer = rangesDf.loc[:, sysName]

                if lowRangeFilter is not None and highRangeFilter is not None:
                    filterSer = pd.qcut(rangesSer, [0, lowRangeFilter, 1-highRangeFilter, 1], labels=False)
                    filterSer.replace(2, 0, inplace=True)

                elif lowRangeFilter is not None:
                    filterSer = pd.qcut(rangesSer, [0, lowRangeFilter, 1], labels=False)

                elif highRangeFilter is not None:
                    filterSer = pd.qcut(rangesSer, [0, 1-highRangeFilter, 1], labels=False)
                    filterSer.replace(1, -1, inplace=True)
                    filterSer.replace(0, 1, inplace=True)
                    filterSer.replace(-1, 0, inplace=True)

                allFilterSers.append(filterSer)

            self.usedWindowsDf = pd.concat(allFilterSers, axis=1)
        else:
            self.usedWindowsDf = None


        # optimization note: this loop is actually quite fast
        for isys, sysName in zip(range(self.xdata.shape[0]), self.allSysNames):
            matSys = self.xdata[isys]
            # shape = w1, w3, t2
            winInd = 0
            for row in range(0, matSys.shape[0] - windowSize + 1, step):
                rstart = row
                rstop = row + windowSize
                for col in range(0, matSys.shape[1] - windowSize + 1, step):
                    winInd += 1
                    if self.usedWindowsDf is not None and self.usedWindowsDf.loc[winInd, sysName] == 0:
                        continue  # skip pre-designated windows
                    cstart = col
                    cstop = col + windowSize
                    window = matSys[rstart:rstop, cstart:cstop, :]
                    norm = np.linalg.norm(window, ord="fro",
                                          axis=(0, 1))  # compute frobenius norm with matrix = axis 0, 1
                    # norm is now a 1d vector with size t2

                    # I go with this simple vector normalization scheme because I don't
                    # necessarily expect the distribution to be gaussian
                    if normalizeVectors:
                        vecRange = rangesDf.loc[winInd, sysName]
                        norm = (norm - min) / vecRange

                    if isys not in sysDict.keys():
                        sysDict[isys] = norm.reshape(1, 250) #TODO: hardcoded to num t2 steps
                    else:
                        sysDict[isys] = np.concatenate((sysDict[isys], norm.reshape(1, 250)), axis=0)


        if self.usedWindowsDf is not None and questionNum != 6 and saveUsedWindows:  # saving Df for dimer only data is generally useless (already saved when asking question 1)
            self.usedWindowsDf.to_csv(f"usedWindows_{RUNLABEL}_train={isTrain}_rangeAlld={lowRangeFilter}-{highRangeFilter}.csv")

        for count, key in enumerate(sysDict):
            if count == 0:
                x = sysDict[key]
            else:
                x = np.concatenate((x, sysDict[key]), axis=0)

        # FINAL x shape = (num training examples, t2 axis = 250)
        #                   windows per sys (self.NUM_WINDOWS) * num sys, 250

        # FINAL y shape = (num training examples, )

        self.nSamples = x.shape[0]
        self.x = torch.from_numpy(x).float()
        self.y = self.setNewY(questionNum)

        if self.x.shape[0] != self.y.shape[0]:
            raise Exception(
                f"Shape of x data points {self.x.shape} is not equal to number of y data points {self.y.shape}")

    def setNewY(self, question):
        """
        this function allows us to update the dataset's labels for answering a new question without reprocessing
        all the x data
        """
        allSysNoPostKey = [el.replace(self.POST_KEY, "") for el in self.allSysNames]

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
            vfCategoryDict = {}
            i = 0
            for n in range(1, maxNumVibrations+1):
                combs = itertools.combinations([1300, 800, 200], n)
                for comb in combs:
                    vfCategoryDict[str(list(comb))] = i
                    i += 1

            for vibFreqDict in vibFreqDicts:
                toAdd = vfCategoryDict[str(list(vibFreqDict.values()))]
                allouts.append(toAdd)
        elif question == 5:
            allouts = []
            for vibFreqDict in vibFreqDicts:
                toAdd = 1 if 800 in vibFreqDict.values() else 0
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

        if self.usedWindowsDf is not None:
            repeatSer = self.usedWindowsDf.sum(axis=0)
            # 1 entry for each system, tells how many times that system is in the dataset (each window)
        else:
            repeatSer = pd.Series(dict(zip(self.allSysNames, [self.NUM_WINDOWS] * len(self.allSysNames))))

        y = []

        for label, sysName in zip(allouts, self.allSysNames):
            for ii in range(repeatSer[sysName]):
                y.append(label)

        y = np.array(y)

        self.y = torch.from_numpy(y).long()
        return self.y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.nSamples


class NeuralNet(nn.Module):

    def __init__(self, inputSize, hiddenSizes, numClasses):
        super(NeuralNet, self).__init__()
        self.layers = list()
        for i in range(len(hiddenSizes) + 1):
            inShape = inputSize if i == 0 else hiddenSizes[i - 1]
            outShape = numClasses if i == len(hiddenSizes) else hiddenSizes[i]
            self.layers.append(nn.Linear(inShape, outShape))
            if i != len(hiddenSizes):
                self.layers.append(nn.ReLU())
        self.layers = nn.ModuleList(self.layers)  # must convert to this special object so NN can do proper backprop

    def forward(self, x):
        for i, operation in enumerate(self.layers):
            out = operation(out if i != 0 else x)
        return out


if __name__ == "__main__":

    ###########################
    ####   Pytorch Zone  ######
    ###########################

    if newData:
        labelsLoc = r"/home/jds199/Projects/ML/Dist1.19Devs/labelsJS8002-9042.csv"  # 9042
    else:
        labelsLoc = r"/home/bs407/2tVectors/OREOS/Dist1.15Devs/JS0-1041JPCL.csv"  # 1041

    rangesLoc = "ASSIGNED IN LOOP"
    postKey = "trim"

    verbose = True
    questions = [1, 6]
    winSizes = [5]
    stepFracs = [1]
    hiddenSizess = [[64]]
    trainTestSplit = 0.8
    normalizeVectors = False
    allowedRanges = [[None, None]] if not newData else [[None, None]]  # TODO: change

    totalRuns = len(questions) * len(winSizes) * len(stepFracs)
    runNum = 0

    for minRange, maxRange in allowedRanges:
        for winSize in winSizes:
            for stepFrac in stepFracs:

                if newData:
                    rangesLoc = None
                else:
                    rangesLoc = f"/home/bs407/2tVectors/JS1041Dataset/Ranges/winSize={winSize}stepFrac={stepFrac}/ranges_post={postKey}_winSize={winSize}.csv"

                step = int(stepFrac * winSize)

                # TODO: work in train/dev/test
                randomSeed = random.randint(1, 100000)
                TrainDs = t2Dataset(labelsLoc, postKey, questions[0], winSize, step, randomSeed, isTrain=True,
                                    split=trainTestSplit,
                                    dimersOnly=questions[0] == 6,
                                    normalizeVectors=normalizeVectors,
                                    lowRangeFilter=minRange,
                                    highRangeFilter=maxRange,
                                    RANGESLOC=rangesLoc)
                TestDs = t2Dataset(labelsLoc, postKey, questions[0], winSize, step, randomSeed, isTrain=False,
                                   split=trainTestSplit,
                                   dimersOnly=questions[0] == 6,
                                   normalizeVectors=normalizeVectors,
                                   lowRangeFilter=minRange,
                                   highRangeFilter=maxRange,
                                   RANGESLOC=rangesLoc)

                for question in questions:
                    for hiddenSizes in hiddenSizess:

                        runNum += 1

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
                            TrainDs = t2Dataset(labelsLoc, postKey, question, winSize, step, randomSeed, isTrain=True,
                                                split=trainTestSplit,
                                                dimersOnly=question == 6,
                                                normalizeVectors=normalizeVectors,
                                                lowRangeFilter=minRange,
                                                highRangeFilter=maxRange,
                                                RANGESLOC=rangesLoc
                                                )
                            TestDs = t2Dataset(labelsLoc, postKey, question, winSize, step, randomSeed, isTrain=False,
                                               split=trainTestSplit,
                                               dimersOnly=question == 6,
                                               normalizeVectors=normalizeVectors,
                                               lowRangeFilter=minRange,
                                               highRangeFilter=maxRange,
                                               RANGESLOC=rangesLoc
                                               )

                        nExamples = len(TestDs) + len(TrainDs)

                        inSize = 250  # t2 timeseries
                        numClasses = -1
                        if question == 6:
                            numClasses = 5
                        elif question == 5:
                            numClasses = 8
                        elif question in [1, 2, 3, 4]:
                            numClasses = 2
                        numEpochs = 100
                        batchSize = TrainDs.NUM_WINDOWS * 10  # one batch at least contains 10 systems
                        lr = 0.0001

                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                        # dataloaders automatically shuffle and split data up into batches of requested size
                        train_loader = DataLoader(dataset=TrainDs, batch_size=batchSize, shuffle=True)
                        test_loader = DataLoader(dataset=TestDs, batch_size=1, shuffle=False)

                        model = NeuralNet(inSize, hiddenSizes, numClasses).to(device)

                        lossFn = nn.CrossEntropyLoss()  # THIS LOSS FUNCTION DOES BOTH ONEHOT ENCODING AND SOFTMAX FOR US
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                        numSteps = len(train_loader)

                        for epoch in range(numEpochs):

                            for i, (input, label) in enumerate(train_loader):
                                # loads in one batch
                                input = input.to(device)
                                label = label.to(device)

                                outs = model(input)
                                # outs shape = batchsize, 2
                                loss = lossFn(outs, label)  # model applies softmax to outs at this point
                                with torch.no_grad():
                                    _, preds = torch.max(outs, dim=1)
                                    acc = torch.sum(preds == label) / label.size(dim=0) * 100.
                                    f1 = f1_score(label, preds, average="macro")

                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                                if verbose and i == 0:
                                    print(f"epoch {epoch}/{numEpochs}, loss = {loss.item()}, acc = {acc}, f1 = {f1}",
                                          flush=True)

                        # testing
                        with torch.no_grad():
                            systemInds = TestDs.indsSelected
                            systemNames = [f"runJS{str(ind).zfill(7)}{postKey}" for ind in systemInds]
                            windowsPerSys = TestDs.NUM_WINDOWS
                            allPreds = list()
                            allLabels = list()
                            # TODO: fix this, way of indexing windows is wrong when we are dropping some
                            misclassDf = pd.DataFrame(
                                data=np.full([len(np.arange(0, windowsPerSys)), len(systemNames)], np.nan, dtype=int),
                                index=np.arange(1, windowsPerSys + 1, dtype=int),
                                columns=systemNames)
                            totalCorrect = 0
                            totalSamples = 0
                            currentSysInd = 0
                            currentWindowInSys = 0
                            for index, (input, label) in enumerate(test_loader):
                                currentWindowInSys += 1
                                if currentWindowInSys == windowsPerSys + 1:
                                    currentSysInd += 1
                                    currentWindowInSys = 1
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

                                if not isCorrect:
                                    currentSys = f"runJS{str(TestDs.indsSelected[currentSysInd]).zfill(7)}{postKey}"
                                    misclassDf.loc[currentWindowInSys, currentSys] = prediction.item()

                            acc = 100.0 * totalCorrect / totalSamples
                            f1 = f1_score(allLabels, allPreds, average="macro")

                            print(f"answering question {question}", flush=True)
                            print(
                                f"NN Specs: 1 Hidden layer sizes {hiddenSizes}, Batch size = {batchSize}, num classes = {numClasses}, lr = {lr}",
                                flush=True)

                            print(
                                f"WinSize={winSize}, Step={step}, Step Fraction={stepFrac}, ranges allowed={minRange}-{maxRange},"
                                f" normalize = {normalizeVectors} \n",
                                flush=True)

                            print(f"Validation Accuracy={acc}, F1 Score = {f1}\n\n", flush=True)

                            outDir = f"/home/bs407/2tVectors/JS1041Dataset/Confusion Histograms/Question {question}/CSVs"

                            if not os.path.exists(outDir):
                                os.makedirs(outDir)

                            misclassDf.to_csv(f"{outDir}/misclass_{RUNLABEL}_{int(time.time())}.csv")
                            # full of test data results, index = window index, col = system name
                            # element = int(np.nan) = -2147483648 if correctly classified and element = predicted if incorrectly classified

                TrainDs = None
                del TrainDs
                TestDs = None
                del TestDs
                train_loader = None
                del train_loader
                test_loader = None
                del test_loader
                gc.collect()

