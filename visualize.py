import numpy as np
import random
from matplotlib import animation
import matplotlib as mpl

import t2vectors
from t2vectors import t2Dataset
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation


#mpl.rcParams["animation.ffmpeg_path"] = r"C:/Users/bmans/ffmpeg/bin/ffmpeg.exe"

question = 0
winSize = 5
step = 5
stepFrac = 1
inds = [51, 200, 470, 749, 900, 1010]  # systems to look at
t2 = 0
labelsLoc = r"/home/bs407/2tVectors/OREOS/Dist1.15Devs/JS0-1041JPCL.csv"
DATALOC = r"/home/jds199/NO_BACKUP/Database_nobackup/Simulated/postprocessed"
postKey = "trim"
rangesLoc = f"/home/bs407/2tVectors/JS1041Dataset/Ranges/winSize={winSize}stepFrac={stepFrac}/ranges_post={postKey}_winSize={winSize}.csv"

lowRangeFilter = .15
highRangeFilter = None


ds = t2Dataset(labelsLoc, postKey, 1, winSize, step, split=1, specificIndicies=inds,
               lowRangeFilter=lowRangeFilter, highRangeFilter=highRangeFilter, RANGESLOC=rangesLoc, saveUsedWindows=False)


for count, sysInd in enumerate(inds):

    # visualize full system pngs
    dir = f"Vector Samples/JS1041PNGs/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.figure()
    mat = ds.xdata[count, :, :, t2]
    plt.matshow(mat, cmap="Spectral")

    w1, w3 = mat.shape
    for i in range(0, w1, 5):
        plt.axhline(i - 0.5, color='white', linewidth=0.5)
    for j in range(0, w3, 5):
        plt.axvline(j - 0.5, color='white', linewidth=0.5)

    winInd = 1
    for i in range(0, 5*(w1//5), 5):
        for j in range(0, 5*(w1//5), 5):
            used = ds.usedWindowsDf.loc[winInd, f"runJS{str(sysInd).zfill(7)}{postKey}"] == 1
            if used:
                plt.fill([j - 0.5, j + 4.5, j + 4.5, j - 0.5, j - 0.5],
                         [i - 0.5, i - 0.5, i + 4.5, i + 4.5, i - 0.5], 'red', alpha=0.3)
            if winInd % 10 == 0 or winInd == 1:
                plt.text(j+2.5, i+2.5, str(winInd), ha='center', va='center', color='white', fontsize=6)
            winInd += 1

    plt.grid(visible=False)


    plt.savefig(f"{dir}System{sysInd},t2={t2},ranges={lowRangeFilter}-{highRangeFilter}.png")

    # visualize t2 vectors

    imagesWritten = 0
    newdir = f"Vector Samples/{winSize}x{winSize}step{step}ind{sysInd}"
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    for i, windowInd in enumerate(ds.usedWindowsDf.loc[:, f"runJS{str(sysInd).zfill(7)}{postKey}"]):
        plt.figure()
        plt.plot(np.array(ds[i][0]))  # select out x data for example i
        plt.savefig(f"{newdir}/{winSize}x{winSize}step{step},sys={sysInd},window={windowInd}")
        plt.close()
        imagesWritten += 1
        if imagesWritten > 2:  # so we dont need to have so many images
            break

    print(f"finished condition: winSize {winSize}, step {step}, ind {sysInd}", flush=True)
