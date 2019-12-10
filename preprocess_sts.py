import os


def append_test(folder):
    goldstandard = ""
    gs_files = ["/gs/STS.gs.answers-forums.txt", "/gs/STS.gs.answers-students.txt",
                "/gs/STS.gs.belief.txt", "/gs/STS.gs.headlines.txt", "/gs/STS.gs.images.txt"]
    for gs in gs_files:
        with open(folder+gs) as data:
            for line in data:
                goldstandard += line
    raw = ""
    raw_files = ["/raw/STS.input.answers-forums.txt", "/raw/STS.input.answers-students.txt",
                "/raw/STS.input.belief.txt", "/raw/STS.input.headlines.txt", "/raw/STS.input.images.txt"]

    raw_gs = ""
    gs_list = goldstandard.split("\n")
    for in_file in raw_files:
        with open(folder+in_file) as data:
            for line in data:
                raw += line
    raw_list = raw.split("\n")
    i=0
    for line in raw_list:
        if gs_list[i] != "":
            raw_gs += "{} \t {} \n".format(gs_list[i],line)
        i+=1
    with open("{}/raw.txt".format(folder), "w+") as output:
        output.write(str(raw))
    with open("{}/gs.txt".format(folder), "w+") as output:
        output.write(str(goldstandard))
    with open("{}/raw_gs.txt".format(folder), "w+") as output:
        output.write(str(raw_gs))


def append_train(folder):
    goldstandard = ""
    gs_files = ["/gs/STS.gs.answers-forum.txt", "/gs/STS.gs.answers-students.txt",
                "/gs/STS.gs.belief.txt", "/gs/STS.gs.headlines.txt", "/gs/STS.gs.images.txt", 
                "/gs/2014_test/STS.gs.deft-forum.txt", "/gs/2014_test/STS.gs.deft-news.txt", 
                "/gs/2014_test/STS.gs.headlines.txt", "/gs/2014_test/STS.gs.images.txt", 
                "/gs/2014_test/STS.gs.OnWN.txt", "/gs/2014_test/STS.gs.tweet-news.txt", 
                "/gs/2013_test/STS.gs.FNWN.txt", "/gs/2013_test/STS.gs.headlines.txt", 
                "/gs/2013_test/STS.gs.OnWN.txt", "/gs/2012_test/STS.gs.MSRpar.txt",
                "/gs/2012_test/STS.gs.MSRvid.txt","/gs/2012_test/STS.gs.SMTeuroparl.txt",
                "/gs/2012_test/STS.gs.surprise.OnWN.txt","/gs/2012_test/STS.gs.surprise.SMTnews.txt"]
    for gs in gs_files:
        with open(folder+gs) as data:
            for line in data:
                goldstandard += line
    raw = ""
    raw_files = ["/raw/STS.input.answers-forum.txt", "/raw/STS.input.answers-students.txt",
                "/raw/STS.input.belief.txt", "/raw/STS.input.headlines.txt", "/raw/STS.input.images.txt",
                 "/raw/2014_test/STS.input.deft-forum.txt", "/raw/2014_test/STS.input.deft-news.txt", 
                 "/raw/2014_test/STS.input.headlines.txt", "/raw/2014_test/STS.input.images.txt", 
                 "/raw/2014_test/STS.input.OnWN.txt", "/raw/2014_test/STS.input.tweet-news.txt", 
                 "/raw/2013_test/STS.input.FNWN.txt", "/raw/2013_test/STS.input.headlines.txt", 
                 "/raw/2013_test/STS.input.OnWN.txt", "/raw/2012_test/STS.input.MSRpar.txt",
                "/raw/2012_test/STS.input.MSRvid.txt","/raw/2012_test/STS.input.SMTeuroparl.txt",
                "/raw/2012_test/STS.input.surprise.OnWN.txt","/raw/2012_test/STS.input.surprise.SMTnews.txt"]
    raw_gs = ""
    gs_list = goldstandard.split("\n")
    for in_file in raw_files:
        with open(folder+in_file) as data:
            for line in data:
                raw += line
    raw_list = raw.split("\n")
    i=0
    for line in raw_list:
        if gs_list[i] != "":
            raw_gs += "{} \t {} \n".format(gs_list[i],line)
        i+=1
    with open("{}/raw.txt".format(folder), "w+") as output:
        output.write(str(raw))
    with open("{}/gs.txt".format(folder), "w+") as output:
        output.write(str(goldstandard))
    with open("{}/raw_gs.txt".format(folder), "w+") as output:
        output.write(str(raw_gs))


if __name__ == "__main__":
    append_test("./data/sts_test")
    append_train("./data/sts_train")
