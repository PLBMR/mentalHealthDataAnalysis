#getGenderCountFrame.py
#helper that gets the list of genders inputted by respondents and counts said
#genders in the process

#imports

import pandas as pd

#helpers

def getGenderCountFrame(rawFrame,genderVarname):
    #helper that builds our gender counter dataframe
    rawFrame["responseID"] = range(rawFrame.shape[0])
    #then groupby
    genderCountFrame = rawFrame.groupby(genderVarname,
            as_index = False)["responseID"].count()
    #rename some information
    genderCountFrame = genderCountFrame.rename(columns = 
                                {genderVarname:"gender","responseID":"count"})
    return genderCountFrame

#main process

if __name__ == "__main__":
    rawFilename_2016 = "../data/raw/osmi-survey-2016_data.csv"
    rawFilename_2014 = "../data/raw/osmi-survey-2014_data.csv"
    rawFrame_2016 = pd.read_csv(rawFilename_2016)
    rawFrame_2014 = pd.read_csv(rawFilename_2014)
    genderCountFrame_2016 = getGenderCountFrame(rawFrame_2016,
                                                "What is your gender?")
    genderCountFrame_2014 = getGenderCountFrame(rawFrame_2014,"Gender")
    genderCountFrame = genderCountFrame_2016.append(genderCountFrame_2014,
                                                    ignore_index = True)
    genderCountFrame.to_csv("../data/preprocessed/genderCountFrame.csv",
                            index = False)
