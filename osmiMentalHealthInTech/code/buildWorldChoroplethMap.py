#buildWorldChoroplethMap.py
#helper script for building a choropleth for observation data for the world

#imports

import plotly.plotly as py
import pandas as pd

#helpers

#for constructiong dataset

def preparePlotlyCountryFrame(plotlyCountryFilename):
    #helper for preparing the plotly country frame
    plotlyFrame = pd.read_csv(plotlyCountryFilename)
    #clear up some information
    plotlyFrame = plotlyFrame.loc[:,["COUNTRY","CODE"]]
    plotlyFrame = plotlyFrame.rename(columns={"COUNTRY":"country",
                                              "CODE":"code"})
    plotlyFrame["country"] = plotlyFrame["country"].str.strip()
    #rename one country for analysis purposes
    plotlyFrame.loc[plotlyFrame["country"] == "United States",
                    "country"] = "United States of America"
    return plotlyFrame

def prepareCountryCountFrame(rawFilename):
    #helper for preparing the frame of counts per country in our main dataset
    rawFrame = pd.read_csv(rawFilename)
    #aggregate by country counts
    rawFrame["responseID"] = range(rawFrame.shape[0])
    countryQString = "What country do you work in?"
    countryCountFrame = rawFrame.groupby(countryQString,as_index = False)[
                                            "responseID"].count()
    #rename some levels
    countryCountFrame = countryCountFrame.rename(columns =  {
                                countryQString:"country","responseID":"count"})
    return countryCountFrame

def makeCanonicalDataset(plotlyFilename,rawFilename):
    #helper that makes our canonical country count frame
    plotlyFrame = preparePlotlyCountryFrame(plotlyFilename)
    countryCountFrame = prepareCountryCountFrame(rawFilename)
    #perform merge
    canonicalFrame = plotlyFrame.merge(countryCountFrame,on = "country",
                                        how = "left")
    #get rid of nulls
    canonicalFrame.loc[canonicalFrame["count"].isnull(),"count"] = 0
    return canonicalFrame

#for plotting dataset

def plotDataset(dataFrame,scl,scaleName,plotName,plotFilename):
    #helper for plotting my dataset
	data = [ dict(
        	type = 'choropleth',
        	locations = dataFrame["code"],
        	z = dataFrame["count"],
                text = dataFrame["country"],
                colorscale = scl,
                autocolorscale = False,
                reversescale = True,
                marker = dict(
                    line = dict (
                        color = 'rgb(180,180,180)',
                        width = 0.5
                    ) ),
                colorbar = dict(
                    autotick = False,
                    title = scaleName),
                ) ]
        layout = dict(
            title = plotName,
            geo = dict(
                showframe = False,
                showcoastlines = False,
                projection = dict(
                    type = 'Mercator'
                )
            )
        )
        #then export
        fig = dict(data=data,layout=layout)
        py.iplot(fig,validate=False,filename=plotFilename )

if __name__ == "__main__":
    #links to raw frame and country frame for plotly
    plotlyCountryFilename = "https://raw.githubusercontent.com/plotly/" \
                            "datasets/master/2014_world_gdp_with_codes.csv"
    rawFilename = "../data/raw/osmi-survey-2016_data.csv"
    #make our dataset
    canonicalFrame = makeCanonicalDataset(plotlyCountryFilename,rawFilename)
    #then plot
    scl = [[0,"rgb(5, 10, 172)"],[0.02,"rgb(40, 60, 190)"],
            [0.15,"rgb(70, 100, 245)"],[0.25,"rgb(90, 120, 245)"],
            [0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]]
    plotDataset(canonicalFrame,scl,"Number of Respondents",
                "Number of Respondents by Country",
                "numRespondentsByCountry")
