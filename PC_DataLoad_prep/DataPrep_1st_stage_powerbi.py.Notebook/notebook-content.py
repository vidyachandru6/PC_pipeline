# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "7a996052-6058-4f64-8259-1c6ec7b26637",
# META       "default_lakehouse_name": "PC_Dataload_prep_Lakehouse",
# META       "default_lakehouse_workspace_id": "866cf7cd-1a96-41ef-b1b8-6989f6f14b47",
# META       "known_lakehouses": [
# META         {
# META           "id": "7a996052-6058-4f64-8259-1c6ec7b26637"
# META         }
# META       ]
# META     },
# META     "warehouse": {
# META       "known_warehouses": []
# META     }
# META   }
# META }

# CELL ********************

# Import pachages
from pyspark.sql.types import *
from pyspark.sql.functions import *
import numpy as np
import pandas as pd
import os
import openpyxl
import pathlib
from pathlib import Path

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

mssparkutils.fs.exists("abfss://PC_DataLoad_prep@onelake.dfs.fabric.microsoft.com/PC_Dataload_prep_Lakehouse.Lakehouse/Files")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#LakeHousePath = "Files/PD_RawData"
LakeHousePath = "abfss://PC_DataLoad_prep@onelake.dfs.fabric.microsoft.com/PC_Dataload_prep_Lakehouse.Lakehouse/Files"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

"""
NTxWthrData_filepath = Path(LakeHousePath)/"DataPrep2021_23.csv"
NTxWthrData = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(str(NTxWthrData_filepath))
"""
NTxWthrData = pd.read_csv("abfss://PC_DataLoad_prep@onelake.dfs.fabric.microsoft.com/PC_Dataload_prep_Lakehouse.Lakehouse/Files/PD_RawData/DataPrep2021_23.csv", header=1)
"""
FlowerMoundData_filepath = Path(LakeHousePath)/"Flower_Mound_2021_3.csv"
FlowerMoundData = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(str(FlowerMoundData_filepath))
"""
FlowerMoundData = pd.read_csv("abfss://PC_DataLoad_prep@onelake.dfs.fabric.microsoft.com/PC_Dataload_prep_Lakehouse.Lakehouse/Files/PD_RawData/Flower_Mound_2021_3.csv", header=2)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import datetime
#from datetime import datetime
today = datetime.date.today()
print(today)
year = today.year
import re

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

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
        """
        print(PollenData[:3])
        # Emulate header=1 behavior because pollendata has column headders
        # in header=1
        PollenData.columns = PollenData.iloc[0]  # Set the first row as the header
        print(PollenData.columns.values)
        PollenData = PollenData[1:].reset_index(drop=True)  # Remove the now-duplicated first row
        """
        #display(PollenData.columns.values)
        PollenData = PollenData[ColsToUse]
        PollenData = PollenData.fillna(0)
        
        PollenData['PollenCnt'] = 0
        for col in ColsToUse:
            if col != 'Date':
                PollenData[col] = PollenData[col].astype(int)
                PollenData['PollenCnt'] = PollenData['PollenCnt']+PollenData[col]
        PollenData['Date'] =  PollenData['Date'].astype(str)

        PollenData['Date'] = PollenData['Date'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d"))

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
    

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#display(FlowerMoundData.columns.values)
stage1prep = dataprep1(FlowerMoundData, NTxWthrData)
PreparedPollenData = stage1prep.prepare_dataset1()
print(PreparedPollenData.columns.values)
MergedData = stage1prep.mergedatasets(PreparedPollenData)
TrainData, ValData =stage1prep.trainvalsplit(MergedData)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

TrainData_sparkDF = spark.createDataFrame(TrainData)
ValData_sparkDF = spark.createDataFrame(ValData)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

Temp_path1 = "Files/Temp/TrainData"
Temp_path2 = "Files/Temp/ValData"
TrainData_sparkDF.coalesce(1).write.mode("overwrite").format("csv").option("header", "true").save(Temp_path1)
ValData_sparkDF.coalesce(1).write.mode("overwrite").format("csv").option("header", "true").save(Temp_path2)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import notebookutils
fs_path2 = Path("Files/Temp/ValData/*.csv")
files_traindata = notebookutils.fs.ls("Files/Temp/TrainData")
files_valdata = notebookutils.fs.ls("Files/Temp/ValData")

for file in files_traindata:
    print(f"Name: {file.name}, Path: {file.path}, Is Directory: {file.isDir}")


"""
import glob
files_traindata = glob.glob("Files/Temp/TrainData/*.csv")
files_valdata = glob.glob("Files/Temp/TrainData/*.csv")
"""

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Create schema if not exists and write to lakehouse
#spark.sql("Create schema if not exists for TrainData and ValData")
# Step 3: Find the part file and move it
import os
Temp_path1 = "Files/Temp/TrainData/"
Temp_path2 = "Files/Temp/Valdata/"
PreppedData_path = "abfss://PC_Model@onelake.dfs.fabric.microsoft.com/PC_Model_Lakehouse.Lakehouse/Files"
for file in files_traindata:
    print(file.name)
    if file.name.endswith(".csv"):
        notebookutils.fs.mv(file.path, "abfss://PC_Model@onelake.dfs.fabric.microsoft.com/PC_Model_Lakehouse.Lakehouse/Files/TrainData.csv", overwrite=True)
for file in files_valdata:
    print(file.name)
    if file.name.endswith(".csv"):
        print(file.name, file.path)
        notebookutils.fs.mv(file.path, "abfss://PC_Model@onelake.dfs.fabric.microsoft.com/PC_Model_Lakehouse.Lakehouse/Files/ValData.csv", overwrite=True)

# Optional: Clean up the temp folder
#notebookutils.fs.rm(os.path.join(Temp_path1,"/TrainData"), recurse=True)
#notebookutils.fs.rm(os.path.join(Temp_path2,"/ValData"), recurse=True)

#Tables_path = "abfss://1eae7a41-187c-448d-b33a-39a7f731ab8f@onelake.dfs.fabric.microsoft.com/16554174-10d3-4fb6-9c88-76c2efb59b51/Tables"

#TrainData_sparkDF.write.mode("overwrite").saveAsTable(f"{Tables_path}/TrainData")
#ValData_sparkDF.write.mode("overwrite").saveAsTable(f"{Tables_path}/ValData")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
