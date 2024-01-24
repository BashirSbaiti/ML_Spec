import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open("Hyperparameter Search/Question 6/Data Params/run1.log", "r") as f:
    dict = {}
    for line in f:
        if line.find("WinSize") != -1:
            cols = list()
            searchSpace = line
            doneSearching = False
            while not doneSearching:
                coli = searchSpace[0:searchSpace.find("=")]
                cols.append(coli)
                if searchSpace.find(",") == -1:
                    vali = searchSpace[searchSpace.find("=") + 1:searchSpace.find(r"\n")]
                else:
                    vali = searchSpace[searchSpace.find("=") + 1:searchSpace.find(",")]

                vali = int(vali) if vali.find(".") == -1 else float(vali)

                if coli in dict.keys():
                    dict[coli].append(vali)
                else:
                    dict[coli] = [vali]

                if searchSpace.find(", ") == -1:
                    doneSearching = True
                else:
                    searchSpace = searchSpace[searchSpace.find(", ") + 2:]

df = pd.DataFrame(dict)
print(df)
x = df["WinSize"]
y = df["Step Fraction"]
z = df["Validation Accuracy"]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x, y, z)
ax.set_xlabel("WinSize")
ax.set_ylabel("Step Fraction")
ax.set_zlabel("Validation Accuracy")
plt.show()
