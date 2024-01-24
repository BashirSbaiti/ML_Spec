import time
import pickle
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
import random
from t2vectors import t2Dataset, loadSystemData

DATALOC = r"/home/jds199/ML_Database/Database/Simulated/postprocessed"
labelsLoc = r"/home/bs407/2tVectors/OREOS/Dist1.15Devs/JS0-1041JPCL.csv"
postKey = "trim"

winSize = 5
stepFrac = 1

randomSeed = random.randint(1, 100000)
fullDs = t2Dataset(labelsLoc, postKey, 0, winSize, winSize*stepFrac, isTrain=True, split=1)
loader = DataLoader(dataset=fullDs, batch_size=1, shuffle=False)

currentSysInd = 0
currentWindowInSys = 0
windowsPerSys = fullDs.NUM_WINDOWS
systemInds = fullDs.indsSelected
systemNames = [f"runJS{str(ind).zfill(7)}{postKey}" for ind in systemInds]

rangesDf = pd.DataFrame(data=np.full([len(np.arange(0, windowsPerSys)), len(systemNames)], 0, dtype=float),
                          index=np.arange(1, windowsPerSys + 1, dtype=int),
                          columns=systemNames)

for index, (input, label) in enumerate(loader):
    currentWindowInSys += 1
    if currentWindowInSys == windowsPerSys + 1:
        currentSysInd += 1
        currentWindowInSys = 1

    range = input.max().item() - input.min().item()

    currentSysName = f"runJS{str(fullDs.indsSelected[currentSysInd]).zfill(7)}{postKey}"
    rangesDf.loc[currentWindowInSys, currentSysName] = range


outDir = f"/home/bs407/2tVectors/JS1041Dataset/Ranges/winSize={winSize}stepFrac={stepFrac}"
if not os.path.exists(outDir):
    os.makedirs(outDir)

rangesDf.to_csv(f"{outDir}/ranges_post={postKey}_winSize={winSize}.csv")


