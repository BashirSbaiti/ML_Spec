import numpy as np
import random
from matplotlib import animation
import matplotlib as mpl

import t2vectors
from t2vectors import t2Dataset
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation

# mpl.rcParams["animation.ffmpeg_path"] = r"C:/Users/bmans/ffmpeg/bin/ffmpeg.exe"

question = 0
winSize = 5
step = 5
stepFrac = 1
inds = [9600]  # systems to look at
t2 = 0
DATASET = "JS9042_11083Dataset"
labelsLoc = r"/home/bs407/2tVectors/JS9042_11083.csv"
XDATALOC = "/home/bs407/2tVectors/JS9042_11083Dataset/xData"
RAWDATALOC = "/home/jds199/ML_Database/Database/Simulated/postprocessed"
postKey = "trim"
rangesLoc = f"/home/bs407/2tVectors/JS9042_11083Dataset/Ranges/ranges_winSize={winSize}_step={step}__{postKey}.csv"

lowRangeFilter = None
highRangeFilter = None

ds = t2Dataset(XDATALOC, labelsLoc, postKey, 1, winSize, step, split=1, specificIndicies=inds,
               lowRangeFilter=lowRangeFilter, highRangeFilter=highRangeFilter, RANGESLOC=rangesLoc,
               saveUsedWindows=False)

for count, sysInd in enumerate(inds):

    # visualize full system pngs
    dir = f"{DATASET}/Vector Samples"
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.figure()
    runName = f"runJS{str(sysInd).zfill(7)}{postKey}"
    mat = t2vectors.loadSystemData(runName, RAWDATALOC)[:, :, t2]
    plt.matshow(mat, cmap="Spectral")

    w1, w3 = mat.shape
    for i in range(0, w1, 5):
        plt.axhline(i - 0.5, color='white', linewidth=0.5)
    for j in range(0, w3, 5):
        plt.axvline(j - 0.5, color='white', linewidth=0.5)

    winInd = 1
    for i in range(0, 5 * (w1 // 5), 5):
        for j in range(0, 5 * (w1 // 5), 5):
            if ds.usedWindowsDf is not None:
                used = ds.usedWindowsDf.loc[winInd, f"runJS{str(sysInd).zfill(7)}{postKey}"] == 1
                if used:
                    plt.fill([j - 0.5, j + 4.5, j + 4.5, j - 0.5, j - 0.5],
                             [i - 0.5, i - 0.5, i + 4.5, i + 4.5, i - 0.5], 'red', alpha=0.3)
            if winInd % 10 == 0 or winInd == 1:
                plt.text(j + 2.5, i + 2.5, str(winInd), ha='center', va='center', color='white', fontsize=6)
            winInd += 1

    plt.grid(visible=False)

    plt.savefig(f"{dir}/System{sysInd},t2={t2},ranges={lowRangeFilter}-{highRangeFilter}.png")

    # visualize t2 vectors

    imagesWritten = 0
    newdir = f"{DATASET}/Vector Samples/{winSize}x{winSize}step{step}ind{sysInd}"
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    if ds.usedWindowsDf is not None:
        winsSer = ds.usedWindowsDf.loc[:, f"runJS{str(sysInd).zfill(7)}{postKey}"]
        winList = winsSer[winsSer == 1].index
    else:
        winList = np.arange(1, ds.NUM_WINDOWS+1)
    for i, windowInd in enumerate(winList):
        plt.figure()
        plt.plot(np.array(ds[i][0]))  # select out x data for example i
        plt.savefig(f"{newdir}/{winSize}x{winSize}step{step},sys={sysInd},window={windowInd}")
        plt.close()
        imagesWritten += 1
        if imagesWritten > 2:  # so we dont need to have so many images
            break

    print(f"finished condition: winSize {winSize}, step {step}, ind {sysInd}", flush=True)
