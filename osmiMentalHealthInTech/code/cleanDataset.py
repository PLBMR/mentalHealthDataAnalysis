#cleanDataset.py
#script written to clean the raw dataset and return to the canonical form for
#the hypotheses and questions asked on data.world

#imports

import pandas as pd
import csv

#helpers

def cutoffAges(dataFrame): #helper that cuts ages of respondents off at a
    #reasonable level to remove the possibility of outliers
    ageCutoff = 100 #not considering ages above 100
    dataFrame = dataFrame[dataFrame["What is your age?"] <= ageCutoff]
    return dataFrame

def processGenderFile(genderMapFilename):
    #helper for processing the file for our gender map and outputing a
    #dictionary of the map
    #open the file
    genderMapFile = open(genderMapFilename,"rb")
    genderMapCSV = csv.reader(genderMapFile)
    #load in data
    genderMapFrame = []
    for row in genderMapCSV:
        genderMapFrame.append(row)
    #then make map
    genderMap = {}
    origGenderInd = 0
    mapGenderInd = 2
    #run through .csv frame
    for i in xrange(1,len(genderMapFrame)):
        genderLev = genderMapFrame[i]
        origGender = genderLev[origGenderInd]
        mappedGender = genderLev[mapGenderInd]
        #then make map
        genderMap[origGender] = mappedGender
    #then return
    return genderMap

def mapGender(dataFrame,genderMap):
    #helper for mapping gender in our data frame
    genderVec = dataFrame["What is your gender?"]
    dataFrame["What is your gender?"] = genderVec.map(genderMap)
    #drop null observations
    dataFrame = dataFrame[dataFrame["What is your gender?"].notnull()]
    return dataFrame

def recodeWorkplaceSize(dataFrame):
    #helper that recodes workplace size to accurate represent those who are
    #self-employed
    employeeString = ("How many employees does your company or "
                        + "organization have?")
    dataFrame.loc[dataFrame[employeeString].isnull(),
                    employeeString] = "Self-Employed"
    return dataFrame

def findRoleType(workString):
    #helper that finds the role type of an individual based on the string of
    #job descriptions provided
    roleDict = {"technical":["Front-end Developer","Back-end Developer",
                         "DevOps/SysAdmin","Designer"],
            "non-technical":["HR","Support","Sales","Dev Evangelist/Advocate",
                             "Other","Excecutive Leadership",
                             "Supervisor/Team Lead","One-person shop"]}
    #run through role dictionary
    for technicalRole in roleDict["technical"]:
        #search to find it in job descriptions
        if (technicalRole in workString):
            #now see if they take on an additional non-technical role
            for nonTechnicalRole in roleDict["non-technical"]:
                if (nonTechnicalRole in workString): #they do both
                    return "both"
            #if we got to this point, they are just technical
            return "technical"
    #if we got here, they are non-technical
    return "non-technical"

def buildRoleType(dataFrame):
    #helper to build the role type variable
    dataFrame["roleType"] = dataFrame[
        "Which of the following best describes your work position?"].apply(
                            findRoleType)
    return dataFrame

def processColNameFile(colNameFilename): #helper for processing our map for
    #column names
    colNameFrame = pd.read_csv(colNameFilename)
    colMap = {} #we will build this
    for i in xrange(colNameFrame.shape[0]):
        #get old name
        oldName = colNameFrame["oldName"].iloc[i]
        newName = colNameFrame["newName"].iloc[i]
        #then make map
        colMap[oldName] = newName
    return colMap

def renameAndPrepareExport(dataFrame,nameDict):
    #prepares our dataset for export
    #rename columns
    dataFrame = dataFrame.rename(columns = nameDict)
    #then export appropriately
    exportCols = nameDict.values()
    dataFrame = dataFrame.loc[:,exportCols]
    return dataFrame

def processData(rawFrame,genderMap,nameDict): #helper that processes our raw 
    #data to become the canonical dataset for modeling purposes
    procFrame = rawFrame
    #first get rid of odd ages
    procFrame = cutoffAges(procFrame)
    #map gender
    procFrame = mapGender(procFrame,genderMap)
    #recode job descriptions
    procFrame = buildRoleType(procFrame)
    #recode size of workplace
    procFrame = recodeWorkplaceSize(procFrame)
    #then rename some columns and set export
    procFrame = renameAndPrepareExport(procFrame,nameDict)
    return procFrame

#main process

if __name__ == "__main__":
    rawDataFilename = "../data/raw/osmi-survey-2016_data.csv"
    genderMapFilename = "../data/preprocessed/genderCountFrame.csv"
    colNameFilename = "../data/preprocessed/nameMap.csv"
    #prepare auxilary componnents
    genderMapDict = processGenderFile(genderMapFilename)
    nameMapDict = processColNameFile(colNameFilename)
    #then run process
    rawDataFrame = pd.read_csv(rawDataFilename)
    procDataFrame = processData(rawDataFrame,genderMapDict,nameMapDict)
    #then export
    procDataFrame.to_csv("../data/processed/procDataset.csv",index = False)
