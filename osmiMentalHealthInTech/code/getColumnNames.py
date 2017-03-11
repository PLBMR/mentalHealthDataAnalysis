#getColumnNames.py
#Quick subroutine to list out all relevant column names of the data frame

#imports

import pandas as pd

#run method

responseFrame = pd.read_csv("../data/raw/osmi-survey-2016_data.csv")
for columnName in responseFrame.columns:
    print columnName
