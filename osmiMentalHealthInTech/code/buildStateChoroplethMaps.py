#buildChoroplethMaps.py
#helper for building our Choropleth maps for the data society challenge and the
#potential blog post

#code currently taken from plot.ly, will edit it to my adjustment

#imports

import plotly.plotly as py
import pandas as pd
import numpy as np

#helpers

#building our dataset

def loadInStateInfo():
    #helper for loading in our state information by code and name
    stateLink = ("https://raw.githubusercontent.com/plotly/datasets/master/"
                 + "2011_us_ag_exports.csv")
    stateAgFrame = pd.read_csv(stateLink)
    #only retain state-level info (i.e. codes, etc.)
    stateFrame = stateAgFrame.loc[:,["code","state"]]
    #clear whitespace
    stateFrame["state"] = stateFrame["state"].str.strip()
    return stateFrame

def getStateAggregates(rawFrame):
    #helper that makes our state aggregates based on raw OSMI survey
    rawFrame["responseID"] = range(rawFrame.shape[0]) #helper for count
    usaFrame = rawFrame[rawFrame["What country do you work in?"]
                            == "United States of America"]
    stateQString = "What US state or territory do you work in?"
    #count state occurences
    stateCountFrame = usaFrame.groupby(stateQString,as_index = False)[
                        "responseID"].count()
    stateCountFrame = stateCountFrame.rename(columns={stateQString:"state",
                                                        "responseID":"numObs"})
    #then count number of diagnoses
    #make map
    diagWithMHDString = ("Have you been diagnosed with a mental health"
                          + " condition by a medical professional?")
    usaFrame["diagWithMHD"] = usaFrame[diagWithMHDString].map({"No":0,"Yes":1})
    #then aggregate
    stateDiagCountFrame = usaFrame.groupby(stateQString,as_index = False)[
                                "diagWithMHD"].sum()
    stateDiagCountFrame = stateDiagCountFrame.rename(columns={
                                                stateQString:"state",
                                                "diagWithMHD":"numDiagnoses"})
    #then return
    return (stateCountFrame,stateDiagCountFrame)

def findPropDiagnosed(row):
    #row helper to find number diagnosed
    if (row["numObs"] == 0):
        return 0
    else:
        return (float(row["numDiagnoses"]) / row["numObs"])

def makeTextSection(stateFrame):
    #helper for making the text section of the state frame
    textStateFrame = stateFrame
    #do some roundoff
    percentMul = 100
    textStateFrame["propDiagnosed"] = np.round(textStateFrame["propDiagnosed"] 
                                        * percentMul)
    for col in textStateFrame.columns:
        textStateFrame[col] = textStateFrame[col].astype(str)
    #then make text sectioni
    textStateFrame["text"] = (textStateFrame["state"] + "<br>"
                + "# Respondents: " + textStateFrame["numObs"] + "<br>"
                + "# Diagnosed with MHD: " + textStateFrame["numDiagnoses"]
                + "<br>"
                + "% Diagnosed: " + textStateFrame["propDiagnosed"])
    stateFrame["text"] = list(textStateFrame["text"])
    return stateFrame

def joinWithAllStates(stateFrame,rawFrame):
    #helper that joins the information aggregated from rwaFrame with information
    #in stateFrame
    #get our state aggregates in rawFrame
    (stateCountFrame,stateDiagCountFrame) = getStateAggregates(rawFrame)
    #perform joins
    stateFrame = stateFrame.merge(stateCountFrame,on = "state",how = "left")
    stateFrame = stateFrame.merge(stateDiagCountFrame,on = "state",how = "left")
    #then clean up columns
    stateFrame.loc[stateFrame["numObs"].isnull(),"numObs"] = 0
    stateFrame.loc[stateFrame["numDiagnoses"].isnull(),"numDiagnoses"] = 0
    #then get prop diagnosed
    stateFrame["propDiagnosed"] = stateFrame.apply(findPropDiagnosed,axis = 1)
    #then make text section
    stateFrame = makeTextSection(stateFrame)
    return stateFrame

#plotting our dataset

def plotStateVar(stateVar,scl,scaleName,plotName,exportName,stateFrame):
    #helper that plots our state variable into a particular layout
    #make data and layout
    data = [ dict(
         type='choropleth',
         colorscale = scl,
         autocolorscale = False,
         locations = stateFrame["code"],
         z = stateFrame[stateVar].astype(float),
         locationmode = 'USA-states',
         text = stateFrame['text'],
         marker = dict(
             line = dict (
                 color = 'rgb(255,255,255)',
                 width = 2
             ) ),
         colorbar = dict(
             title = scaleName)
         )]
    layout = dict(
         title = plotName,
         geo = dict(
         scope='usa',
         projection=dict( type='albers usa' ),
         showlakes = True,
         lakecolor = 'rgb(255, 255, 255)'),
        )
    #then export
    fig = dict( data=data, layout=layout )
    py.iplot(fig,filename = exportName)

#main process

if __name__ == "__main__":
    #get state listings
    stateFrame = loadInStateInfo()
    #then get raw data
    print stateFrame.columns
    rawFrame = pd.read_csv("../data/raw/osmi-survey-2016_data.csv")
    #process our state information with our raw data
    stateFrame = joinWithAllStates(stateFrame,rawFrame)
    #then plot
    #6 steps from yellow to blue
    scl = [[0.0, 'rgb(255,255,0)'],[0.2, 'rgb(204,204,51)'],
            [0.4, 'rgb(153,153,102)'],[0.6, 'rgb(102,102,153)'],
            [0.8, 'rgb(51,51,204)'],[1.0, 'rgb(0,0,255)']]
    #for num obs
    plotStateVar("numObs",scl,"# of Respondents",
                 "Number of Respondents by State","numRespondentsByState",
                 stateFrame)
    #for num diagnoses
    plotStateVar("numDiagnoses",scl,"# of Diagnoses",
                 "Number of Respondents Diagnosed with Mental Health Disorder"
                 + "<br>By State",
                 "numDiagnosesByState",stateFrame)
    #for prop diagnoses
    plotStateVar("propDiagnosed",scl,"Proportion of Diagnoses",
                 "Proportion of Respondents Diagnosed with Mental Health "
                 + "Disorder<br>By State",
                 "propDiagnosesByState",stateFrame)
# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv')
# 
# for col in df.columns:
#     df[col] = df[col].astype(str)
# print df["code"]
# 
# 
# df['text'] = df['state'] + '<br>' +\
#     'Beef '+df['beef']+' Dairy '+df['dairy']+'<br>'+\
#     'Fruits '+df['total fruits']+' Veggies ' + df['total veggies']+'<br>'+\
#     'Wheat '+df['wheat']+' Corn '+df['corn']
# 

# 
# layout = dict(
#         title = '2011 US Agriculture Exports by State<br>(Hover for breakdown)',
#         geo = dict(
#             scope='usa',
#             projection=dict( type='albers usa' ),
#             showlakes = True,
#             lakecolor = 'rgb(255, 255, 255)'),
#              )
#     
# fig = dict( data=data, layout=layout )
# py.iplot(fig,filename = "d3-choropleth-map")
