"""
Generate a dict of key system name key window index value t2 vector
:param LABELSLOC: location of labels dataframe csv file (output from extract_labels.py) IMPORTANT note this will also
        implicitly set what system indexes (within DATALOC) are being looked at through the keys of the dataframe
:param POST_KEY: appended string to runname, indicating which postprocessing training data has undergone
:param winSize: window size to make t2 vectors
:param step: step size between windows
:param normalizeVectors: weather to normalize vectors or not. normalization is carried out by subtracting by the
minimum value and dividing by the range (max-min value)
"""
import pickle
import time
import os
import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessingPool
import pathos

POST_KEY = "trim"
windowSize = 5
step = 5
normalizeVectors = False
saveRanges = True

saveLabel = f"winSize={windowSize}_step={step}_{'norm' if normalizeVectors else ''}_{POST_KEY}"
dsName = "JS9042Dataset"

DATALOC = r"/home/jds199/ML_Database/Database/Simulated/postprocessed"  # new data 8002-9042
LABELSLOC = r"/home/jds199/Projects/ML/Dist1.20Devs/JS8002_9042.csv"

sysinfoDf = pd.read_csv(LABELSLOC, index_col=0)
allSystemNames = list(sysinfoDf.index)


def loadSystemData(name, DATALOC):
    # loads system with specific run name from the directory DATALOC
    # returns: (1, w1, w3, t2) array that describes the system

    fullpath = f"{DATALOC}/{name}.pkl"

    with open(fullpath, "rb") as pklfile:
        fullEntry = pickle.load(pklfile)
        idata = fullEntry['R']["Rw1w3Absorptive"]

    return idata

    # shape of xdata = (num systems, w1, w3, t2)


start = time.perf_counter()
numCores = pathos.helpers.cpu_count()
print(f"Detected {numCores} CPUs", flush=True)
pool = ProcessingPool(nodes=numCores)
with pool as executor:
    allSystemNamesPost = [f"{name}{POST_KEY}" for name in allSystemNames]
    print(f"beginning multiprocess at {start} sec", flush=True)
    results = executor.map(loadSystemData, allSystemNamesPost, [DATALOC] * len(allSystemNamesPost))
    # results ends up stacked in such a way that it is equivalent to concatenating the systems on axis 0
    # so the shape ends up being NUM_SYSTEMS, w1, w3, t2 without any np reshape or concat
    allSpectra = np.array(results)
    print(f"ending multiprocess after {time.perf_counter() - start} sec", flush=True)

# allSpectra shape: NUM_SYSTEMS, w1, w3, t2

# chopping up the data

windowsPerSys = int(allSpectra.shape[1] / windowSize) ** 2
rangesDf = pd.DataFrame(data=np.full([len(np.arange(0, windowsPerSys)), len(allSystemNames)], 0, dtype=float),
                        index=np.arange(1, windowsPerSys + 1, dtype=int),
                        columns=allSystemNamesPost)

xDict = {}  # dict of dict, key1=system name (w/post key), key2=windowIndex, value=t2Vector

for isys, sysName in zip(range(allSpectra.shape[0]), allSystemNamesPost):
    matSys = allSpectra[isys]
    # shape = w1, w3, t2
    winInd = 0
    xDict[sysName] = {}
    for row in range(0, matSys.shape[0] - windowSize + 1, step):
        rstart = row
        rstop = row + windowSize
        for col in range(0, matSys.shape[1] - windowSize + 1, step):
            winInd += 1
            cstart = col
            cstop = col + windowSize
            window = matSys[rstart:rstop, cstart:cstop, :]
            norm = np.linalg.norm(window, ord="fro",
                                  axis=(0, 1))  # compute frobenius norm with matrix = axis 0, 1
            # norm is now a 1d vector with size t2

            # I go with this simple vector normalization scheme because I don't
            # necessarily expect the distribution to be gaussian
            if normalizeVectors or saveRanges:
                minVal = np.min(norm)
                maxVal = np.max(norm)

                vecRange = maxVal - minVal
                rangesDf.loc[winInd, sysName] = vecRange

                norm = (norm - minVal) / vecRange if normalizeVectors else norm

            xDict[sysName][winInd] = norm.astype(np.float16)

savePath = f"/home/bs407/2tVectors/{dsName}"

savePathxData = f"{savePath}/xData"
if not os.path.exists(savePathxData):
    os.makedirs(savePathxData)

fullPath = f"{savePathxData}/xData_{saveLabel}.pkl"
with open(fullPath, "wb") as pklfile:
    pickle.dump(xDict, pklfile)

if saveRanges:
    savePathRanges = f"{savePath}/Ranges"
    if not os.path.exists(savePathRanges):
        os.makedirs(savePathRanges)

    fullPath = f"{savePathRanges}/ranges_{saveLabel}.csv"
    rangesDf.to_csv(fullPath)
