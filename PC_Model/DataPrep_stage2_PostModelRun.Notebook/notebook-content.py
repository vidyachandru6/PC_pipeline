# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   }
# META }

# CELL ********************

# %%
# Connect to the lib packages

# %%
import numpy as np
from numpy import array

from numpy.linalg import norm
import pickle
import pandas as pd
import os
import math
import matplotlib.pyplot as plt

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************



# %%
class Dataprep_preModelInput():
    def __init__(self, *args):
        if len(args)>0:
            self.datafile = args[0]
    def get_ModelInputData(self):    
        # Read the training data vis-a-vis input and output training data from the input dataframe
        
        x1 = np.vstack(self.datafile['Day']).astype(int)
        x2 = np.vstack(self.datafile['Month']).astype(int)
        x3 = np.vstack(self.datafile['MAX']).astype(np.float32)
        x4 = np.vstack(self.datafile['MIN']).astype(np.float32)
        x5 = np.vstack(self.datafile['AVG']).astype(np.float32)
        x6 = np.vstack(self.datafile['WTR']).astype(np.float32)
        x7 = np.vstack(self.datafile['AVGSPD']).astype(np.float32)
        x8 = np.vstack(self.datafile['MAXSPD']).astype(np.float32)
        y = np.vstack(self.datafile['PollenCnt']).astype(int)
        
        return x1, x2, x3,x4,x5,x6,x7,x8,y
    def normalize(self, x1, x2, x3, x4,x5,x6,x7,x8,y):
        if x1.std()!= 0.0:
            x1_N = (x1-x1.mean())/x1.std() 
        else: x1_N = x
        if x2.std()!= 0.0:
            x2_N = (x2-x2.mean())/x2.std() 
        else: x2_N = x2
        x3_trueMean = x3.mean()
        x3_trueSD = x3.std()
        if x3_trueSD!= 0.0:
            x3_N = (x3-x3_trueMean)/x3_trueSD 
        else: x3_N = x3
        
        if x4.std()!= 0.0:
            x4_N = (x4-x4.mean())/x4.std() 
        else: x4_N = x4
        
        if x3_trueSD!= 0.0:
            x3_N = (x3-x3_trueMean)/x3_trueSD 
        else: x3_N = x3

        if x5.std()!= 0.0:
            x5_N = (x5-x5.mean())/x5.std() 
        else: x5_N = x5
        if x6.std()!= 0.0:
            x6_N = (x6-x6.mean())/x6.std() 
        else: x6_N = x6

        if x7.std()!= 0.0:
            x7_N = (x7-x7.mean())/x7.std() 
        else: x7_N = x7

        if x8.std()!= 0.0:
            x8_N = (x8-x8.mean())/x8.std() 
        else: x8_N = x8
        
        y_trueMean = y.mean()
        y_trueSD = y.std()
        if y_trueSD!=0:
            y_N = (y-y_trueMean)/y_trueSD
        else:
            y_N = y
        
        
        print("Inside Normalize function: mean: "+str(y_trueMean)+" SD: "+str(y_trueSD))
        return x1_N, x2_N, x3_N, x4_N,x5_N,x6_N,x7_N, x8_N,y_N, y_trueMean, y_trueSD
    
    def get_trainval_tensor_dataset(self, x1,x2, x3_N, x4_N, x5_N,x6_N,x7_N, x8_N,y_N):
        # COnverting training data and testing data to torch tensor
        x_train = np.column_stack((x1,x2, x3_N, x4_N, x5_N,x6_N,x7_N, x8_N))
        y_train = y_N
        y_train = np.reshape(y_train,[-1])
        train_x = torch.tensor(x_train, dtype=torch.float)
        print(train_x.shape)
        print(train_x.shape[-1])
        train_y_N = torch.tensor(y_train, dtype=torch.float) 
        print(train_y_N.shape)
        
        #test_y = torch.tensor(y_train, dtype = torch.float) # training data is testing data
        #print(test_y.size())
        return train_x, train_y_N
    def get_test_tensor_dataset(self,x1,x2, x3_N, x4_N, x5_N,x6_N,x7_N, x8_N):
        # COnverting training data and testing data to torch tensor
        x_train = np.column_stack((x1,x2, x3_N, x4_N, x5_N,x6_N,x7_N, x8_N))
        train_x = torch.tensor(x_train, dtype=torch.float)
        print(train_x.shape)
        print(train_x.shape[-1])
        return train_x
    def DataProcess(self):
        x_Month, x_Day,x_MaxTemp, x_MinTemp, x_AvgTemp,x_Wtr,x_WndAvgSpd, x_WndMaxSpd, y = self.get_ModelInputData() 
        x_Month_N, x_Day_N,x_MaxTemp_N, x_MinTemp_N, x_AvgTemp_N,x_Wtr_N,x_WndAvgSpd_N, x_WndMaxSpd_N, y_N, y_TrueMean, y_TrueSD = self.normalize(x_Month, x_Day,x_MaxTemp, x_MinTemp, x_AvgTemp,x_Wtr,x_WndAvgSpd, x_WndMaxSpd, y)
        NData = len(self.datafile)
        print("Number of data is: "+str(NData))
        trtest_x, trtest_y = self.get_trainval_tensor_dataset(x_Month_N, x_Day_N,x_MaxTemp_N, x_MinTemp_N, x_AvgTemp_N,x_Wtr_N,x_WndAvgSpd_N, x_WndMaxSpd_N, y_N)
        return trtest_x, trtest_y, y_TrueMean, y_TrueSD, NData


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

class PostModelrun_DataPrep():
    def __init__(self, *args):
        if len(args)>0:
            self.ModelInput = args[0]
            self.ModelMean = args[1]
            self.ModelVar = args[2]
            self.norm = args[3]
    def PredDataProcess(self):
        ModelInput = self.ModelInput
        ModelMean = self.ModelMean
        ModelVar = self.ModelVar
        PollenCount_preds = pd.DataFrame(columns=['Year','Month','Day','MaxTemp','MinTemp','AvgTemp','Rain','WindAvgSpd',\
                                                'WindMaxSpd','ActualPollenCount','PredictedPollenCount', 'PredictedPollenCountSD'])
        #ColsOrder = ['Day','Month','Year','MAX','MIN','AVG','WTR','AVGSPD','MAXSPD','PollenCnt']
        cols=list(ModelInput.columns.values)
        print(cols)
        PollenCount_preds['Year'] = ModelInput['Year']
        PollenCount_preds['Month']= ModelInput['Month']
        PollenCount_preds['Day'] = ModelInput['Day']
        PollenCount_preds['MaxTemp'] = ModelInput['MAX']
        PollenCount_preds['MinTemp'] = ModelInput['MIN']
        PollenCount_preds['AvgTemp'] = ModelInput['AVG']
        PollenCount_preds['Rain'] = ModelInput['WTR']
        PollenCount_preds['WindAvgSpd'] = ModelInput['AVGSPD']
        PollenCount_preds['WindMaxSpd'] = ModelInput['MAXSPD']
        PollenCount_preds['ActualPollenCount'] = ModelInput['PollenCnt']
        print(ModelMean.shape)
        Mean = ModelMean.numpy()
        Var = ModelVar.numpy()
        print(Mean[:10])
        PollenCount_preds['PredictedPollenCount'] = Mean
        #Var = np.squeeze(Var)
        PollenCount_preds['PredictedPollenCountSD'] = Var
        cols=list(PollenCount_preds.columns.values)
        print(cols)
        return PollenCount_preds
    def numpy_outputs_plots(self,num_epochs, path):
        mean_np =self.ModelMean.numpy()
        var_np = self.ModelVar.numpy()
        mean_np = np.squeeze(mean_np)
        var_np = np.squeeze(var_np)
        true_output = np.squeeze(self.ModelInput['PollenCnt'])
        self.ModelInput['Year'] = self.ModelInput['Year'].astype(int)
        yr_min = self.ModelInput['Year'].max()
        yr_max = self.ModelInput['Year'].max()
        yr = str(yr_min)+" to "+str(yr_max)
        N = len(self.ModelMean)
        print("The model output shape of mean and var are respectively :")
        print(mean_np.shape, var_np.shape, true_output.shape)
        print(mean_np.ndim)
        title = "Test data vs model pollen count predictions  for "+str(yr)+' with L2Norm '+str(norm)+' '
        _,ax = plt.subplots(1,1, figsize=(9,6), sharex=True, sharey = True)
        x_axis = np.linspace(0, N, N)
        #print(mean_np[:20], true_output[:20])
        ax.set(ylabel="Pollen count output", title=title+str(num_epochs)+" iterations")# with L2Norm "+str(norm))
        
        
        ax.plot(x_axis, true_output,color='red',alpha=0.5,lw=3)#label='Training Points')
        ax.plot(x_axis, mean_np,color='C4', alpha=0.5, label="Predictive mean")
        ax.fill_between(
            x_axis,
            (mean_np - (2*(np.abs(var_np)))**0.5),
            (mean_np + (2*(np.abs(var_np)))**0.5),
            #color="#CDC0B0",
            color="cyan",
            #color="gray",
            alpha=0.4,
            lw=1.5,
        )
        ax.title.set_text(title+" with "+str(num_epochs)+" iterations")
        ax.legend(["Actual","Predicted"], loc="upper right")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(os.path.join(path,"Pollen Count output with L2Norm "+str(norm)+".jpg"))
        plt.show()
        return mean_np, var_np


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
