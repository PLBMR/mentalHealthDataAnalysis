#prepareDatasetForClustering.r
#helper script to process our raw dataset for clustering purposes

#imports

#helpers

processRawData = function(rawFrame,colMapFrame){
    #helper that processes our raw dataset to prepare for clustering
    #clear out self-employed individuals
    rawFrame = rawFrame[which(rawFrame$Are.you.self.employed. != 1),]
    #then rename columns
    newColNameVec = c()
    deleteColInds = c()
    for (i in 1:dim(colMapFrame)[1]){
        #get colMapFrame row
        colMapRow = colMapFrame[i,]
        if (colMapRow$newColName == ""){ #don't need this column for cluster
            deleteColInds = c(deleteColInds,i)
        }
        else { #add to new column names
            newColNameVec = c(newColNameVec,as.character(colMapRow$newColName))
        }
    }
    #delete certain columns
    deleteColumnNames = colnames(rawFrame)[deleteColInds]
    rawFrame = rawFrame[,!names(rawFrame) %in% deleteColumnNames]
    #then rename our columns
    colnames(rawFrame) = newColNameVec
    return(rawFrame)
}

#main process

#prepare our files
rawFilename = "../data/raw/osmi-survey-2016_data.csv"
procFilename = "../data/processed/clusterDataset.csv"
mapFilename = "../data/preprocessed/clusterColumnMap.csv"
#load in our data
rawFrame = read.csv(rawFilename)
colMapFrame = read.csv(mapFilename)
#process our data
procFrame = processRawData(rawFrame,colMapFrame)
#then export
write.csv(procFrame,procFilename,row.names = FALSE)
