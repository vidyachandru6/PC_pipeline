# %%
# Connect to the lib packages
import sys
sys.path.append(r"C:\Users\vidya\BNN\Lib\site-packages")
sys.path.append(r"C:\Users\vidya\AppData\Local\Programs\Python\Python310\lib\site-packages")

# %%
import pandas as pd
import os

# %%
SRCFILES = "./Data/RawData/"#C:/Users/vidya/OneDrive/Documents/Vidya/Fall 2025/dfwwthr/"
PATH = "./TrainedModel/"#C:/Users/vidya/OneDrive/Documents/Vidya/Fall 2025/dfwwthr/TrainedModels/"
SRCFILES = "C:/Users/vidya/OneDrive/Documents/Vidya/Fall 2025/dfwwthr/"

# %%
NTxWthrData = pd.read_csv(os.path.join(SRCFILES+"DataPrep2021_23.csv"))
FlowerMoundData = pd.read_csv(os.path.join(SRCFILES+"Flower_Mound_2021_3.csv"))

# %%
import datetime
#from datetime import datetime
today = datetime.date.today()
print(today)
year = today.year
import re

# %%
class dataprep1():
    def __init__(self, *args):
        if len(args)>0:
            self.dataset1 = args[0]
            self.dataset2 = args[1]
    def prepare_dataset1(self):
        ColsToUse = ['Date','Acer','Alnus','Ambrosia','Arecaceae','Artemisia','Asteraceae (Excluding Ambrosia and Artemisia)','Betula',\
             'Carpinus/Ostrya','Carya','Celtis','Chenopodiaceae/Amaranthaceae ','Corylus','Cupressaceae','Cyperaceae',\
             'Eupatorium','Fagus','Fraxinus','Gramineae / Poaceae','Juglans','Ligustrum','Liquidambar','Morus','Olea',\
             'Other Grass Pollen','Other Tree Pollen','Other Weed Pollen','Pinaceae','Plantago','Platanus','Populus',\
             'Prosopis','Quercus','Rumex','Salix','Tilia','Ulmus','Unidentified Pollen','Urticaceae']

            
        PollenData = self.dataset1
        
        # Emulate header=1 behavior because pollendata has column headders
        # in header=1
        PollenData.columns = PollenData.iloc[0]  # Set the first row as the header
        PollenData = PollenData[1:].reset_index(drop=True)  # Remove the now-duplicated first row

        PollenData = PollenData[ColsToUse]
        PollenData = PollenData.fillna(0)
        
        PollenData['PollenCnt'] = 0
        for col in ColsToUse:
            if col != 'Date':
                PollenData[col] = PollenData[col].astype(int)
                PollenData['PollenCnt'] = PollenData['PollenCnt']+PollenData[col]
        PollenData['Date'] =  PollenData['Date'].astype(str)

        PollenData['Date'] = PollenData['Date'].apply(lambda x: datetime.datetime.strptime(x,"%m/%d/%Y"))

        PollenData['Year'] = PollenData['Date'].apply(lambda x: x.year)
        PollenData['Month'] = PollenData['Date'].apply(lambda x: x.month)
        PollenData['Day'] = PollenData['Date'].apply(lambda x: x.day)
        PollenData['Year'] = PollenData['Year'].astype(int)
        PollenData['Month'] = PollenData['Month'].astype(int)
        PollenData['Day'] = PollenData['Day'].astype(int)
        ColsToUse.remove('Date')
        PollenData.drop(ColsToUse, axis=1, inplace=True)
        print(PollenData.columns.values)
        return PollenData
    
    def mergedatasets(self, data1):
        print(data1.columns.values)
        WthrDataCols = ['MNTH', 'Day', 'Year', 'MAX', 'MIN', 'AVG', 'WTR', 'AVGSPD', 'MAXSPD']
        self.dataset2 = self.dataset2[WthrDataCols]
        print(self.dataset2.columns.values)
        PreppedData = pd.merge(data1, self.dataset2, how='inner', left_on=['Year','Month','Day'], right_on=['Year','MNTH','Day'], )
        print(list(PreppedData.columns.values))
        ColsOrder = ['Day','Month','Year','MAX','MIN','AVG','WTR','AVGSPD','MAXSPD','PollenCnt']
        PreppedData = PreppedData[ColsOrder]
        PreppedData['MIN'] = PreppedData['MIN'].fillna(0)
        PreppedData['MAX'] = PreppedData['MAX'].fillna(0)
        PreppedData['AVG'] = PreppedData['AVG'].fillna(0)
        PreppedData['WTR'] = PreppedData['WTR'].fillna(0)
        PreppedData['AVGSPD'] = PreppedData['AVGSPD'].fillna(0)
        PreppedData['MAXSPD'] = PreppedData['MAXSPD'].fillna(0)
        PreppedData['PollenCnt'] = PreppedData['PollenCnt'].fillna(0)
        PreppedData['MIN'] = PreppedData['MIN'].astype(float)
        PreppedData['MAX'] = PreppedData['MAX'].astype(float)
        PreppedData['AVG'] = PreppedData['AVG'].astype(float)
        PreppedData['WTR'] = PreppedData['WTR'].astype(float)
        PreppedData['AVGSPD'] = PreppedData['AVGSPD'].astype(float)
        PreppedData['MAXSPD'] = PreppedData['MAXSPD'].astype(float)
        PreppedData['PollenCnt'] = PreppedData['PollenCnt'].astype(int)
        return PreppedData
    
    def trainvalsplit(self,mergeddata):
        TrainingData = mergeddata[mergeddata['Year']<=2022]
        ValidationData = mergeddata[mergeddata['Year']==2023]
        return TrainingData, ValidationData
    


# %%
stage1prep = dataprep1(FlowerMoundData, NTxWthrData)
PreparedPollenData = stage1prep.prepare_dataset1()
print(PreparedPollenData.columns.values)
MergedData = stage1prep.mergedatasets(PreparedPollenData)
TrainData, ValData =stage1prep.trainvalsplit(MergedData)



