#cleanDataset.py
#script written to clean the raw dataset and return to the canonical form for
#the hypotheses and questions asked on data.world

#imports

import pandas as pd
import csv

#helpers

def cutoffAges(dataFrame,ageVarname): #helper that cuts ages of respondents off 
    #at a reasonable level to remove the possibility of outliers
    ageCutoff = 100 #not considering ages above 100
    dataFrame = dataFrame[dataFrame[ageVarname] <= ageCutoff]
    dataFrame = dataFrame[dataFrame[ageVarname] > 0]
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

def mapGender(dataFrame,genderMap,genderVarname):
    #helper for mapping gender in our data frame
    genderVec = dataFrame[genderVarname]
    dataFrame[genderVarname] = genderVec.map(genderMap)
    #drop null observations
    dataFrame = dataFrame[dataFrame[genderVarname].notnull()]
    return dataFrame

def recodeWorkplaceSize(dataFrame):
    #helper that recodes workplace size to accurate represent those who are
    #self-employed
    employeeString = ("How many employees does your company or "
                        + "organization have?")
    dataFrame.loc[dataFrame[employeeString].isnull(),
                    employeeString] = "1-5" #essentially a self-employment
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

def processColNameFile(colNameFilename,yearConsidered): #helper for processing 
    #our map for column names
    colNameFrame = pd.read_csv(colNameFilename)
    colMap = {} #we will build this
    for i in xrange(colNameFrame.shape[0]):
        #get old name
        oldName = colNameFrame["oldName_" + str(yearConsidered)].iloc[i]
        newName = colNameFrame["newName"].iloc[i]
        #then make map
        if (oldName != "Not Used"):
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

def encodeDiagnosis(procFrame,diagnosisEncodeVarname):
    #helper for encoding the diagnosedWithMHD in the procFrame
    procFrame["diagnosedWithMHD"] = "No"
    procFrame.loc[procFrame[diagnosisEncodeVarname] == "Yes",
                  "diagnosedWithMHD"] = "Yes"
    return procFrame

def processData(rawFrame,genderMap,nameDict,ageVarname,genderVarname,
                roleTypeNeeded = True,diagnosisEncodeVarname = ""): 
    #helper that processes our raw data to become the canonical dataset for 
    #a given year
    procFrame = rawFrame
    #first get rid of odd ages
    if (diagnosisEncodeVarname != ""):
        procFrame = encodeDiagnosis(procFrame,
                                    diagnosisEncodeVarname)#need to encode this
    procFrame = cutoffAges(procFrame,ageVarname)
    #map gender
    procFrame = mapGender(procFrame,genderMap,genderVarname)
    #recode job descriptions
    if (roleTypeNeeded):
        procFrame = buildRoleType(procFrame)
    #recode size of workplace
    procFrame = recodeWorkplaceSize(procFrame)
    #then rename some columns and set export
    procFrame = renameAndPrepareExport(procFrame,nameDict)
    return procFrame

def mergeDataFrames(dataFrameOne,dataFrameTwo):
    #helper to  merge two of our dataframes together appropriately
    #get set intersection of columns
    sharedColumns = set(dataFrameOne.columns) & set(dataFrameTwo.columns)
    slicedDFOne = dataFrameOne[list(sharedColumns)]
    slicedDFTwo = dataFrameTwo[list(sharedColumns)]
    #then append each
    procFrame = slicedDFOne.append(slicedDFTwo,ignore_index = True)
    return procFrame

#main process

if __name__ == "__main__":
    rawDataFilename = "../data/raw/osmi-survey-2016_data.csv"
    genderMapFilename = "../data/preprocessed/genderCountFrame_2016.csv"
    colNameFilename = "../data/preprocessed/nameMap.csv"
    #prepare auxilary componnents
    genderMapDict = processGenderFile(genderMapFilename)
    yearConsidered = 2016
    nameMapDict = processColNameFile(colNameFilename,yearConsidered)
    #then run process
    rawDataFrame = pd.read_csv(rawDataFilename)
    procDataFrame = processData(rawDataFrame,genderMapDict,nameMapDict,
                                "What is your age?","What is your gender?")
    #then export
    procDataFrame.to_csv("../data/processed/procDataset.csv",index = False)
