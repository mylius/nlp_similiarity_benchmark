import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_json("./data/results.json", encoding="utf8")
df = df[["alg", "db", "pearson", "spearman", "mse", "traintime", "runtime"]]

values = {}
values["pear"] = {}
values["spear"] = {}
values["mse"] = {}
values["train"] = {}
values["run"] = {}
for item in df["db"].unique():
    values["pear"][item] = list(df["pearson"].loc[df["db"] == item])
    values["spear"][item] = list(df["spearman"].loc[df["db"] == item])
    values["mse"][item] = list(df["mse"].loc[df["db"] == item])
    values["train"][item] = list(df["traintime"].loc[df["db"] == item])
    values["run"][item] = list(df["runtime"].loc[df["db"] == item])



def create_grouped_bar_graph(sick,sts, title,filename):

    # mseman
    bar_cos_sick = [sick[0],sick[1],sick[2],sick[3],sick[12],sick[13],0,sick[15]]
    bar_l2_sick = [sick[8],sick[9],sick[10],sick[11],0,0,0,0]
    bar_jac_sick = [sick[4],sick[5],sick[6],sick[7],0,0,0,0]
    bar_wmd_sick = [0,0,0,0,0,0,sick[14],0]
    bar_cos_sts = [sts[0],sts[1],sts[2],sts[3],sts[12],sts[13],0,sts[14]]
    bar_l2_sts = [sts[8],sts[9],sts[10],sts[11],0,0,0,0]
    bar_jac_sts = [sts[4],sts[5],sts[6],sts[7],0,0,0,0]
    bar_wmd_sts = [0,0,0,0,0,0,sts[14],0]
    # Make the plot
    barWidth = 0.2
    r1 = np.arange(len(bar_cos_sick))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    plt.figure(figsize=(16,9))
    plt.bar(r1, bar_cos_sick, color='teal', width=barWidth, edgecolor='white', label='cosine')
    plt.bar(r2, bar_l2_sick, color='darkorange', width=barWidth, edgecolor='white', label='l2')
    plt.bar(r3, bar_jac_sick, color='darkred', width=barWidth, edgecolor='white', label='jaccard')
    plt.bar(r4, bar_wmd_sick, color='lightblue', width=barWidth, edgecolor='white', label='wmd')
    
    # Add xticks on the middle of the group bars
    plt.xlabel("STS "+title, fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bar_cos_sick))], ["bow re","bow re stop","bow lem","bow lem stop","spacy W2V","spacy bert","gensim wmd","gensim D2V"])
    plt.savefig('./figs/sick'+filename, dpi = 150)

    plt.figure(figsize=(16,9))
    plt.bar(r1, bar_cos_sts, color='teal', width=barWidth, edgecolor='white', label='cosine')
    plt.bar(r2, bar_l2_sts, color='darkorange', width=barWidth, edgecolor='white', label='l2')
    plt.bar(r3, bar_jac_sts, color='darkred', width=barWidth, edgecolor='white', label='jaccard')
    plt.bar(r4, bar_wmd_sts, color='lightblue', width=barWidth, edgecolor='white', label='wmd')
    
    # Add xticks on the middle of the group bars
    plt.xlabel("STS "+title, fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bar_cos_sick))], ["bow re","bow re stop","bow lem","bow lem stop","spacy W2V","spacy bert","gensim wmd","gensim D2V"])
    plt.savefig('./figs/sts'+filename, dpi = 150)


for item in values:
    entries = []
    for dataset in values[item]:
        entries.append(values[item][dataset])
    if item == "pear":
        title = " – Pearson coefficient"
        name = "_pear.png"
    elif item == "spear":
        title = " – Spearman coefficient"
        name = "_spear.png"
    elif item == "mse":
        title = " – mean squared error"
        name = "_mse.png"
    elif item == "train":
        title = " – training time"
        name = "_train.png"
    elif item == "run":
        title = " – run time"
        name = "_run.png"
    create_grouped_bar_graph(entries[0],entries[1],title,name)
