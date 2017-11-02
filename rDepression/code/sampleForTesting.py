#sampleForTesting.py
#a quick script designed to take a sample of threads and then export them for
#use in our discourse analysis

#imports

import pandas as pd
import pickle as pkl
import sys
import random

#helpers

def sampleThreads(threadFrame,commentFrame,numObsToSample):
    #helper to sample our threads
    #first get our list of sample rows for threads
    threadRows = list(threadFrame.index)
    sampledThreadRows = random.sample(threadRows,numObsToSample)
    sampledThreads = threadFrame.loc[sampledThreadRows,:]
    #then get their respective comments
    linkIDs = list(sampledThreads["id"])
    linkIDs = ["t3_" + linkID for linkID in linkIDs]
    #then grab sampled comments
    sampledComments = commentFrame[commentFrame["link_id"].isin(linkIDs)]
    #then export
    exportDict = {}
    exportDict["posts"] = sampledThreads
    exportDict["comments"] = sampledComments
    return exportDict

#main process

if __name__ == "__main__":
    numObsToSample = int(sys.argv[1])
    #get our data frames ready
    threadFrame = pd.read_csv("../data/raw/allRDepressionPosts.csv")
    commentFrame = pd.read_csv("../data/raw/allRDepressionComments.csv")
    threadDict = sampleThreads(threadFrame,commentFrame,numObsToSample)
    pkl.dump(threadDict,open("../data/preprocessed/testingThreadDict.pkl","wb"))
