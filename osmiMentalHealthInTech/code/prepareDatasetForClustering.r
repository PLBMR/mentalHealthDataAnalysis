#prepareDatasetForClustering.r
#helper script to process our raw dataset for clustering purposes

#imports

#helpers

makeDiscreteEncodings = function(rawFrame,exceptionVec){
    #helper to make discrete encodings for all columns except the ones in
    #exceptionVec
    #first make directory for encoding
    desiredDir = "../data/preprocessed/discreteEncodings"
    if (!dir.exists(desiredDir)){ #make it
        dir.create(desiredDir)
    }
    for (colName in colnames(rawFrame)){
        #check if in exception
        if (!colName %in% exceptionVec){
            #can get unique levels
            levels = unique(rawFrame[,colName])
            #make encoding
            rawFrame[,colName] = match(rawFrame[,colName],
                                      unique(rawFrame[,colName]))
            #then make encoding frame
            encodingFrame = data.frame(level = levels,
                                       encoding = 1:length(levels))
            encodingFilename = paste0(desiredDir,"/",colName,".csv")
            write.csv(encodingFrame,encodingFilename,row.names = FALSE)
        }
    }
    return(rawFrame)
}

processRawData = function(rawFrame,colMapFrame,exceptionVec){
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
    #then re-encode
    rawFrame = makeDiscreteEncodings(rawFrame,exceptionVec)
    return(rawFrame)
}

#main process

#prepare our files
rawFilename = "../data/raw/osmi-survey-2016_data.csv"
procFilename = "../data/processed/clusterDataset.csv"
mapFilename = "../data/preprocessed/clusterColumnMap.csv"
noDiscreteEncodingFilename = "../data/preprocessed/noDiscEncoding.txt"
#load in our data
rawFrame = read.csv(rawFilename)
colMapFrame = read.csv(mapFilename)
exceptionVec = scan(noDiscreteEncodingFilename,what = character())
#process our data
procFrame = processRawData(rawFrame,colMapFrame,exceptionVec)
#then export
write.csv(procFrame,procFilename,row.names = FALSE)
