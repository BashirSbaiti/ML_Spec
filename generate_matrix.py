"""
Generate a dict of key system name value w1, w3, t2 matrix
:param LABELSLOC: location of labels dataframe csv file (output from extract_labels.py) IMPORTANT note this will also
        implicitly set what system indexes (within DATALOC) are being looked at through the keys of the dataframe
:param POST_KEY: appended string to runname, indicating which postprocessing training data has undergone
"""
import gc
import pickle
import time
import os
import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessingPool
import pathos

datakey = "Rw1w3Rephasing"

def loadSystemData(name, DATALOC):
    # loads system with specific run name from the directory DATALOC
    # datakey specifies Rw1w3Absorptive, Rw1w3Rephasing, Rw1w3NonRephasing,...
    # returns: (nChannels, w1, w3, t2) array that describes the system. nChannels is 2 (re, im) if complex, otherwise 1

    fullpath = f"{DATALOC}/{name}.pkl"

    with open(fullpath, "rb") as pklfile:
        fullEntry = pickle.load(pklfile)
        if datakey == "Rw1w3Absorptive":
            idata = np.array(fullEntry['R'][datakey]).astype(np.float16)
            idata = idata[np.newaxis, :]  # one channel
        elif datakey == "Rw1w3Rephasing":
            cidata = np.sum([fullEntry['R']['Rw1w3']['2'], fullEntry['R']['Rw1w3']['3']], axis=0).astype(np.csingle)
            reidata = np.real(cidata).astype(np.float16)[np.newaxis, :]
            imidata = np.imag(cidata).astype(np.float16)[np.newaxis, :]
            idata = np.concatenate([reidata, imidata])  # store separate re and im channels
        elif datakey == "Rw1w3NonRephasing":
            cidata = np.sum([fullEntry['R']['Rw1w3']['1'], fullEntry['R']['Rw1w3']['4']], axis=0).astype(np.csingle)
            reidata = np.real(cidata).astype(np.float16)[np.newaxis, :]
            imidata = np.imag(cidata).astype(np.float16)[np.newaxis, :]
            idata = np.concatenate([reidata, imidata])

    return idata

if __name__ == "__main__":

    POST_KEY = "trim"

    saveLabel = f"w1w3t3Matrix_{POST_KEY}_{datakey}_f16"
    dsName = "JS9042_11083Dataset"

    DATALOC = r"/home/jds199/ML_Database/Database/Simulated/postprocessed"
    LABELSLOC = r"JS9042_11083.csv"

    sysinfoDf = pd.read_csv(LABELSLOC, index_col=0)
    allSystemNames = list(sysinfoDf.index)

    startTime = time.time()
    print(f"Beginning {saveLabel} at {startTime}", flush=True)

    start = time.perf_counter()
    numCores = pathos.helpers.cpu_count()
    print(f"Detected {numCores} CPUs", flush=True)
    pool = ProcessingPool(nodes=numCores)
    with pool as executor:
        allSystemNamesPost = [f"{name}{POST_KEY}" for name in allSystemNames]
        print(f"beginning multiprocess at {start} sec", flush=True)
        allSpectra = np.array(executor.map(loadSystemData, allSystemNamesPost, [DATALOC] * len(allSystemNamesPost))).astype(np.float16)
        # results ends up stacked in such a way that it is equivalent to concatenating the systems on axis 0
        # so the shape ends up being NUM_SYSTEMS, w1, w3, t2 without any np reshape or concat

        print(f"ending multiprocess after {time.perf_counter() - start} sec", flush=True)

    # allSpectra shape: NUM_SYSTEMS, nChannels, w1, w3, t2

    # assign data to system that produced it
    xDict = {allSystemNamesPost[i]: allSpectra[i] for i in range(allSpectra.shape[0])}
    # this makes a dict of key = system name, value = nChannels, w1, w3, t2 to return

    savePath = f"{dsName}"

    savePathxData = f"{savePath}/xData"
    if not os.path.exists(savePathxData):
        os.makedirs(savePathxData)

    fullPath = f"{savePathxData}/xData_{saveLabel}.pkl"
    with open(fullPath, "wb") as pklfile:
        pickle.dump(xDict, pklfile)

    print(f"Ending {saveLabel} after {round(time.time()-startTime, 2)} sec.", flush=True)
