import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_json("./data/results.json",encoding="utf8")
df = df[["alg","db","pearson","spearman","mre","traintime","runtime"]]

sts = df.loc[df["db"] == "sts"]
sick = df.loc[df["db"] == "sick"]
sts = sts.set_index("alg")
sick = sick.set_index("alg")
print(sts)
sts_axes = sts.plot.bar(rot = 0, subplots=True)
for ax1 in sts_axes:
    ax1.plot()
sick_axes = sick.plot.bar(rot = 0, subplots=True)
for ax2 in sick_axes:
    ax2.plot()
plt.show()