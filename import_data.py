import pandas as pd

def importation_file(pathtofile):
    data=pd.read_csv(pathtofile)
    return data
