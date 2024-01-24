syscounts = {}
with open("job2420591.log") as f:
    for line in f:
        if line.find("run") != -1:
            sysname = line[line.find("run")+5:]
            if sysname not in syscounts.keys():
                syscounts[sysname] = 1
            else:
                syscounts[sysname] += 1

for key, val in syscounts.items():
    if val!=2:
        print(key)

print(len(syscounts))
