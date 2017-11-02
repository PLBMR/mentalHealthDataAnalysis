#preprocess.py
#script for filtering our dataset and tokenizing our sentences

#imports

import pandas as pd
import json
from nltk.tokenize import word_tokenize as wt

#helpers

def processData(rawFrame):
    #main process for processing our dataset
    #first, simplify our analysis to just the relevant variables
    relVars = ["body","id","link_id","parent_id"]
    relVarFrame = rawFrame[relVars]
    #then filter out Null or deleted comments
    filteredFrame = relVarFrame[relVarFrame["body"].notnull()]
    filteredFrame = filteredFrame[~(filteredFrame["body"].isin([
                                    "[removed]","[deleted]"]))]
    #tokenize sentences
    sentDict = {row[1]["id"]:wt(row[1]["body"].decode("utf8")) 
                for row in filteredFrame.iterrows()}
    #prepare idFrame
    idFrame = filteredFrame[["id","link_id","parent_id"]]
    return idFrame, sentDict

#main process

if __name__ == "__main__":
    rawCommentFrame = pd.read_csv("../data/raw/allRDepressionComments.csv")
    idFrame, sentDict = processData(rawCommentFrame)
    idFrame.to_csv("../data/preprocessed/idFrame.csv",index = False)
    indentNum = 4
    json.dump(sentDict,open("../data/preprocessed/sentDict.json","wb"),
              indent = indentNum)
