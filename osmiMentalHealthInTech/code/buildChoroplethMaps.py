#buildChoroplethMaps.py
#helper for building our Choropleth maps for the data society challenge and the
#potential blog post

#code currently taken from plot.ly, will edit it to my adjustment

#imports

import plotly.plotly as py
import pandas as pd

#helpers

#building our dataset

def loadInStateInfo():
    #helper for loading in our state information by code and name
    stateLink = ("https://raw.githubusercontent.com/plotly/datasets/master/"
                 + "2011_us_ag_exports.csv")
    stateAgFrame = pd.read_csv(stateLink)
    #only retain state-level info (i.e. codes, etc.)
    stateFrame = stateAgFrame.loc[:,["code","state"]]
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
    return stateFrame

#plotting our dataset

#main process

if __name__ == "__main__":
    #get state listings
    stateFrame = loadInStateInfo()
    #then get raw data
    print stateFrame.columns
    rawFrame = pd.read_csv("../data/raw/osmi-survey-2016_data.csv")
    stateFrame = joinWithAllStates(stateFrame,rawFrame)
    print stateFrame

# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv')
# 
# for col in df.columns:
#     df[col] = df[col].astype(str)
# print df["code"]
# 
# scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
#             [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]
# 
# df['text'] = df['state'] + '<br>' +\
#     'Beef '+df['beef']+' Dairy '+df['dairy']+'<br>'+\
#     'Fruits '+df['total fruits']+' Veggies ' + df['total veggies']+'<br>'+\
#     'Wheat '+df['wheat']+' Corn '+df['corn']
# 
# data = [ dict(
#         type='choropleth',
#         colorscale = scl,
#         autocolorscale = False,
#         locations = df['code'],
#         z = df['total exports'].astype(float),
#         locationmode = 'USA-states',
#         text = df['text'],
#         marker = dict(
#             line = dict (
#                 color = 'rgb(255,255,255)',
#                 width = 2
#             ) ),
#         colorbar = dict(
#             title = "Millions USD")
#         ) ]
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
