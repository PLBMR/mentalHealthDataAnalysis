#getGenderCountFrame.py
#helper that gets the list of genders inputted by respondents and counts said
#genders in the process

#imports

import pandas as pd

#helpers

def getGenderCountFrame(rawFrame):
    #helper that builds our gender counter dataframe
    rawFrame["responseID"] = range(rawFrame.shape[0])
    #then groupby
    genderCountFrame = rawFrame.groupby("What is your gender?",
            as_index = False)["responseID"].count()
    #rename some information
    genderCountFrame = genderCountFrame.rename(columns = {"responseID":"count"})
    return genderCountFrame

#main process

if __name__ == "__main__":
    rawFilename = "../data/raw/osmi-survey-2016_data.csv"
    rawFrame = pd.read_csv(rawFilename)
    genderCountFrame = getGenderCountFrame(rawFrame)
    genderCountFrame.to_csv("../data/preprocessed/genderCountFrame.csv",
                            index = False)
