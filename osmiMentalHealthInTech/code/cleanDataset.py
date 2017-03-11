#cleanDataset.py
#script written to clean the raw dataset and return to the canonical form for
#the hypotheses and questions asked on data.world

#imports

import pandas as pd

#helpers

def cutoffAges(dataFrame): #helper that cuts ages of respondents off at a
    #reasonable level to remove the possibility of outliers
    ageCutoff = 100 #not considering ages above 100
    dataFrame = dataFrame[dataFrame["What is your age?"] <= ageCutoff]
    return dataFrame

def processData(rawFrame): #helper that processes our raw data to become the
    #canonical dataset for modeling purposes
    procFrame = rawFrame
    #first get rid of odd ages
    procFrame = cutoffAges(procFrame)
    return procFrame

#main process

if __name__ == "__main__":
    rawDataFilename = "../data/raw/osmi-survey-2016_data.csv"
    rawDataFrame = pd.read_csv(rawDataFilename)
    procDataFrame = processData(rawDataFrame)
    procDataFrame.to_csv("../data/processed/procDataset.csv",index = False)
