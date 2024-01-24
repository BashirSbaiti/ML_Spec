import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("filePath", help="full file path to directory with log file")
parser.add_argument("fileName", help="name of log file to get loss numbers from")

args = parser.parse_args()

lossLst = list()
filePath = args.filePath
fileName = args.fileName
head = 116

with open(filePath+'/'+fileName) as f:
    for i, line in enumerate(f):
        if line.find("loss = ") != -1 and i+1 > head:
            loss = line[line.find("loss = ")+7:line.find(", acc = ")]
            loss = float(loss)
            lossLst.append(loss)

plt.figure()
plt.plot(lossLst)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.savefig(fileName[0:fileName.find('.')]+"loss.png")

