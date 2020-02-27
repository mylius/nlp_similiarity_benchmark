import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_json("./data/results_all.json", encoding="utf8")
df = df[["alg", "db", "pearson", "spearman", "mse", "traintime", "runtime"]]

sts = df.loc[df["db"] == "sts"]
sick = df.loc[df["db"] == "sick"]
sts = sts.set_index("alg")
sick = sick.set_index("alg")

pear_sick = list(df["pearson"].loc[df["db"] == "sick"])
spear_sick = list(df["spearman"].loc[df["db"] == "sick"])
mse_sick = list(df["mse"].loc[df["db"] == "sick"])
train_sick = list(df["traintime"].loc[df["db"] == "sick"])
run_sick = list(df["runtime"].loc[df["db"] == "sick"])
pear_sts = list(df["pearson"].loc[df["db"] == "sts"])
spear_sts = list(df["spearman"].loc[df["db"] == "sts"])
mse_sts = list(df["mse"].loc[df["db"] == "sts"])
train_sts = list(df["traintime"].loc[df["db"] == "sts"])
run_sts = list(df["runtime"].loc[df["db"] == "sts"])

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

create_grouped_bar_graph(pear_sick,pear_sts," – Pearson coefficient","_pear.png")
create_grouped_bar_graph(spear_sick,spear_sts," – Spearman coefficient","_spear.png")
create_grouped_bar_graph(mse_sick,mse_sts," – mean squared error","_mse.png")
create_grouped_bar_graph(train_sick,train_sts," – training time","_train.png")
create_grouped_bar_graph(run_sick,run_sts," – run time","_run.png")
