#buildTimeDependentDataset.py
#helper for building our dataset that uses time as a relevant variable
#uses cleanDataset.py for most references

#imports

import pandas as pd
from cleanDataset import *

#helpers

#main process

if __name__ == "__main__":
    #get our two datasets ready
    raw2016Filename = "../data/raw/osmi-survey-2016_data.csv"
    raw2014Filename = "../data/raw/osmi-survey-2014_data.csv"
    genderMapFilename = "../data/preprocessed/genderCountFrame.csv"
    colNameFilename = "../data/preprocessed/nameMap.csv"
    #load in our files
    raw2016Frame = pd.read_csv(raw2016Filename)
    raw2014Frame = pd.read_csv(raw2014Filename)
    genderMapDict = processGenderFile(genderMapFilename)
    nameMapDict_2016 = processColNameFile(colNameFilename,2016)
    nameMapDict_2014 = processColNameFile(colNameFilename,2014)
    #process each file
    proc2016Frame = processData(raw2016Frame,genderMapDict,nameMapDict_2016,
                                "What is your age?","What is your gender?")
    proc2014Frame = processData(raw2014Frame,genderMapDict,nameMapDict_2014,
                                "Age","Gender",False,
                    "Have you sought treatment for a mental health condition?")
    #then merge
    proc2016Frame["year"] = 2016
    proc2014Frame["year"] = 2014
    procFrame = mergeDataFrames(proc2016Frame,proc2014Frame)
    procFrame.to_csv("../data/processed/procTimeDataset.csv",index = False)

