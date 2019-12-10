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
    gs_files = ["/raw/STS.input.answers-forums.txt", "/raw/STS.input.answers-students.txt",
                "/raw/STS.input.belief.txt", "/raw/STS.input.headlines.txt", "/raw/STS.input.images.txt"]
    for in_file in gs_files:
        with open(folder+in_file) as data:
            for line in data:
                raw += line
    with open("{}/raw.txt".format(folder), "w+") as output:
        output.write(str(raw))
    with open("{}/gs.txt".format(folder), "w+") as output:
        output.write(str(goldstandard))


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
    gs_files = ["/raw/STS.input.answers-forum.txt", "/raw/STS.input.answers-students.txt",
                "/raw/STS.input.belief.txt", "/raw/STS.input.headlines.txt", "/raw/STS.input.images.txt",
                 "/raw/2014_test/STS.input.deft-forum.txt", "/raw/2014_test/STS.input.deft-news.txt", 
                 "/raw/2014_test/STS.input.headlines.txt", "/raw/2014_test/STS.input.images.txt", 
                 "/raw/2014_test/STS.input.OnWN.txt", "/raw/2014_test/STS.input.tweet-news.txt", 
                 "/raw/2013_test/STS.input.FNWN.txt", "/raw/2013_test/STS.input.headlines.txt", 
                 "/raw/2013_test/STS.input.OnWN.txt", "/raw/2012_test/STS.input.MSRpar.txt",
                "/raw/2012_test/STS.input.MSRvid.txt","/raw/2012_test/STS.input.SMTeuroparl.txt",
                "/raw/2012_test/STS.input.surprise.OnWN.txt","/raw/2012_test/STS.input.surprise.SMTnews.txt"]
    for in_file in gs_files:
        with open(folder+in_file) as data:
            for line in data:
                raw += line
    with open("{}/raw.txt".format(folder), "w+") as output:
        output.write(str(raw))
    with open("{}/gs.txt".format(folder), "w+") as output:
        output.write(str(goldstandard))


if __name__ == "__main__":
    append_test("./data/sts_test")
    append_train("./data/sts_train")
