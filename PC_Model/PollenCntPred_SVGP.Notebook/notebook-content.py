# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "7ef539ed-c049-41a4-a6bb-e73e6e3d443c",
# META       "default_lakehouse_name": "PC_Model_Lakehouse",
# META       "default_lakehouse_workspace_id": "86fb9f96-5262-44fc-81c6-733c42b9dba5",
# META       "known_lakehouses": [
# META         {
# META           "id": "16554174-10d3-4fb6-9c88-76c2efb59b51"
# META         },
# META         {
# META           "id": "7ef539ed-c049-41a4-a6bb-e73e6e3d443c"
# META         },
# META         {
# META           "id": "bbc3e9b3-6023-4f89-be22-616233839653"
# META         },
# META         {
# META           "id": "7a996052-6058-4f64-8259-1c6ec7b26637"
# META         }
# META       ]
# META     },
# META     "environment": {}
# META   }
# META }

# CELL ********************

%pip install gpytorch
%pip install tensorflow
%pip install tensorflow-datasets

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#import packages
from pyspark.sql.types import *
from pyspark.sql.functions import *
import tqdm
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
import math

import numpy as np
from numpy import array

from numpy.linalg import norm
import joblib
import pandas as pd
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
import gpytorch
from torch.nn import Linear
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.kernels import RBFKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, \
    LMCVariationalStrategy
from gpytorch.variational import MeanFieldVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.models.deep_gps.dspp import DSPPLayer, DSPP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import DeepPredictiveLogLikelihood
import gpytorch.settings as settings
#import os
#os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import plot_model
from tensorflow.keras.optimizers import RMSprop
import tensorflow_datasets as tfds
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.optimizers import SGD
# Make plots inline

from scipy.spatial import distance_matrix
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt
%matplotlib inline
import pickle



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import datetime
today = datetime.date.today()
year = today.year

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Get subpackages for Two layer DSPP and 1 layer deepGP models
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from torch.utils.data import TensorDataset, DataLoader

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

scalefactor=3.0

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

class DSPPHiddenLayer(DSPPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=300, inducing_points=None, mean_type='constant', Q=8):
        if inducing_points is not None and output_dims is not None and inducing_points.dim() == 2:
            # The inducing points were passed in, but the shape doesn't match the number of GPs in this layer.
            # Let's assume we wanted to use the same inducing point initialization for each GP in the layer,
            # and expand the inducing points to match this.
            inducing_points = inducing_points.unsqueeze(0).expand((output_dims,) + inducing_points.shape)
            inducing_points = inducing_points.clone() + 0.01 * torch.randn_like(inducing_points)
        if inducing_points is None:
            # No inducing points were specified, let's just initialize them randomly.
            if output_dims is None:
                # An output_dims of None implies there is only one GP in this layer
                # (e.g., the last layer for univariate regression).
                inducing_points = torch.randn(num_inducing, input_dims)
            else:
                inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        else:
            # Get the number of inducing points from the ones passed in.
            num_inducing = inducing_points.size(-2)

        # Let's use mean field / diagonal covariance structure.
        variational_distribution = MeanFieldVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=torch.Size([output_dims]) if output_dims is not None else torch.Size([])
        )

        # Standard variational inference.
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        batch_shape = torch.Size([]) if output_dims is None else torch.Size([output_dims])

        super(DSPPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims, Q)

        if mean_type == 'constant':
            # We'll use a constant mean for the final output layer.
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        elif mean_type == 'linear':
            # As in Salimbeni et al. 2017, we find that using a linear mean for the hidden layer improves performance.
            self.mean_module = LinearMean(input_dims, batch_shape=batch_shape)

        self.covar_module = ScaleKernel(MaternKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
                                        batch_shape=batch_shape, ard_num_dims=None)

    def forward(self, x, mean_input=None, **kwargs):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

class TwoLayerDSPP(DSPP):
    def __init__(self, train_x_shape, inducing_points, num_inducing, hidden_dim=4, Q=3):
        hidden_layer = DSPPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=hidden_dim,
            mean_type='linear',
            inducing_points=inducing_points,
            Q=Q,
        )
        last_layer = DSPPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            mean_type='constant',
            inducing_points=None,
            num_inducing=num_inducing,
            Q=Q,
        )

        likelihood = GaussianLikelihood()

        super().__init__(Q)
        self.likelihood = likelihood
        self.last_layer = last_layer
        self.hidden_layer = hidden_layer

    def forward(self, inputs, **kwargs):
        hidden_rep1 = self.hidden_layer(inputs, **kwargs)
        output = self.last_layer(hidden_rep1, **kwargs)
        return output
    def predict(self,model, test_x):
        with torch.no_grad():

            # The output of the model is a multitask MVN, where both the data points
            # and the tasks are jointly distributed
            # To compute the marginal predictive NLL of each data point,
            # we will call `to_data_independent_dist`,
            # which removes the data cross-covariance terms from the distribution.
            preds = model.likelihood(model(test_x)).to_data_independent_dist()
            #print("Output of the Exact GP model is :"+str(preds.shape()))

        return preds.mean.mean(0), preds.variance.mean(0)
    

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

scalefactor=3.0

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def train_and_test_DSPP(path, train_x,train_y,test_x, test_y, N, num_epochs, y_mean, y_SD):
    # DataLoader is a utility tool in the torch package used
    # to handle minibatches
    train_x = torch.squeeze(train_x)
    train_y =  torch.squeeze(train_y)
    
    test_x = torch.squeeze(test_x)
    test_y = torch.squeeze(test_y)
    train_dataset = TensorDataset(train_x, train_y)
    
    #if y_trueSD == 0: y_trueSD = 0.001
      # this is for running the notebook in our testing framework
    smoke_test = ('CI' in os.environ)
    
    batch_size = 365                 # Size of minibatch
    milestones = [20, 150, 300]       # Epochs at which we will lower the learning rate by a factor of 0.1
    num_inducing_pts = N           # Number of inducing points in each hidden layer
    #num_epochs = 100                  # Number of epochs to train for
    initial_lr = 0.001                 # Initial learning rate
    hidden_dim = 4                    # Number of GPs (i.e., the width) in the hidden layer.
    num_quadrature_sites = 8          # Number of quadrature sites (see paper for a description of this. 5-10 generally works well).
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Use k-means to initialize inducing points (only helpful for the first layer)
    inducing_points = (train_x[torch.randperm(min(1000 * 100, N))[0:num_inducing_pts], :])
    inducing_points = inducing_points.clone().data.cpu().numpy()
    inducing_points = torch.tensor(kmeans2(train_x.data.cpu().numpy(),
                                   inducing_points, minit='matrix')[0])
    if torch.cuda.is_available():
        inducing_points = inducing_points.cuda()
    ## Modified settings for smoke test purposes
    num_epochs = num_epochs if not smoke_test else 1
    model = TwoLayerDSPP(
    train_x.shape,
    inducing_points,
    num_inducing=num_inducing_pts,
    hidden_dim=hidden_dim,
    Q=num_quadrature_sites
    )
    if torch.cuda.is_available():
        model.cuda()

    model.train()
    adam = torch.optim.Adam([{'params': model.parameters()}], lr=initial_lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.MultiStepLR(adam, milestones=milestones, gamma=0.1)


    # The "beta" parameter here corresponds to \beta_{reg} from the paper, and represents a scaling factor on the KL divergence
    # portion of the loss.
    objective = DeepPredictiveLogLikelihood(model.likelihood, model, num_data=N, beta=0.05)
  
    epochs_iter = tqdm_notebook(range(num_epochs), desc="Epoch")

    for i in epochs_iter:
        minibatch_iter = tqdm_notebook(train_loader, desc="Minibatch", leave=False)
        for x_batch, y_batch in minibatch_iter:
            adam.zero_grad()
            output = model(x_batch)
            loss = -objective(output, y_batch)
            loss.backward()
            adam.step()
        sched.step()
    
    # Testing
    model.eval()
    means = torch.tensor([0.])
    
    var = torch.tensor([0.])
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.beta_features.checkpoint_kernel(5):
        for x_batch, y_batch in test_loader:
            means_batch, var_batch = model.predict(model,x_batch)
            means = torch.cat([means, means_batch])
            var = torch.cat([var, var_batch])
    means = means[1:]
    var = var[1:]
    weights = model.quad_weights.unsqueeze(-1).exp().cpu()
    print("Inside Train DSPP")
    print("Mean of training target (Pollen Count): "+str(y_mean)+", SD of training target (Pollen Count): "+str(y_SD))
    means = (means*y_SD)+y_mean
    var = (var*y_SD)+y_mean
    # `means` currently contains the predictive output from each Gaussian in the mixture.
    # To get the total mean output, we take a weighted sum of these means over the quadrature weights.
    rmse = ((weights * means).sum(0) - test_y.cpu()).pow(2.0).mean().sqrt().item()
    
    DSPP_2Norm = torch.linalg.norm(means - test_y.cpu())
    filename = path+'NTXWTHR_DSPP_2Layers.dump'
    joblib.dump(model, open(filename, 'wb'))  
    #torch.save(model.state_dict(), path)
    
   
    return (means, var,DSPP_2Norm)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def test_DSPP(path, test_x, YTest_raw, y_mean, y_SD):
    test_x = torch.squeeze(test_x)
    NTest = test_x.shape[0]
    
    print("Inside test_DSPP: "+str(NTest))
    print(test_x.shape)
    test_y = torch.zeros(NTest)
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    print("Loading from path: "+str(path))
    #model = joblib.load(open(path +'Sunflower_DSPP_2layers_sigmoid_rehospitalization.dump',"rb"))
    #except FileNotFoundError:
    #print("Model does not exist")
    with open(str(path) + "NTXWTHR_DSPP_2Layers.dump","rb") as input_file:
        model = joblib.load(input_file)
    model.eval()
    
    # `means` currently contains the predictive output from each Gaussian in the mixture.
    # To get the total mean output, we take a weighted sum of these means over the quadrature weights.
    means = torch.tensor([0.])
    #means = means.unsqueeze(0)
    var = torch.tensor([0.])
    #var = var.unsqueeze(0)
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.beta_features.checkpoint_kernel(10):
        for x_batch, y_batch in test_loader:
            means_batch, var_batch = model.predict(model,x_batch)
            means = torch.cat([means, means_batch])
            var = torch.cat([var, var_batch])
    means = means[1:]
    var = var[1:]
    mean_np = means.numpy()
    var_np = var.numpy()
    print("Inside Test DSPP")
    print("Mean of training target (Pollen Count): "+str(y_mean)+", SD of training target (Pollen Count): "+str(y_SD))
    mean_np = (mean_np*y_SD)+y_mean
    var_np = (var_np*y_SD)+y_mean
    
    print(" the shape of true test output of is :"+str(YTest_raw.shape))
    print(" the shape of predicted test output of is : "+str(mean_np.shape))
    arr = np.zeros(NTest)
    for i in range(NTest):
        arr[i]=(YTest_raw[i])-(mean_np[i])
    Pred_norm = norm(arr,2)
    #means = torch.exp(means[1:])-1
    #var = torch.exp(var[1:])-1
    
    return (means, var, Pred_norm)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

class OrthDecoupledApproximateGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points,train_x):
        variational_distribution = gpytorch.variational.DeltaVariationalDistribution(inducing_points.size(-2))
        variational_strategy = make_orthogonal_vs(self, train_x)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

class DGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, linear_mean=True):
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        batch_shape = torch.Size([output_dims])
        print("Input features :"+str(input_dims))
        print("Output features :"+str(output_dims))
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super().__init__(variational_strategy, input_dims, output_dims)
        self.mean_module = LinearMean(input_dims) if linear_mean else ConstantMean()
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class MultitaskDeepGP(DeepGP):
    def __init__(self, train_x_shape, train_y_shape,num_tasks):
        gp_layer = DGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims= train_y_shape[-1],
            linear_mean=True
        )

        super().__init__()

        self.gp_layer = gp_layer

        # We're going to use a multitask likelihood instead of the standard GaussianLikelihood
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def forward(self, inputs):
        output = self.gp_layer(inputs)
        return output

    def predict(self,model, test_x):
        with torch.no_grad():

            # The output of the model is a multitask MVN, where both the data points
            # and the tasks are jointly distributed
            # To compute the marginal predictive NLL of each data point,
            # we will call `to_data_independent_dist`,
            # which removes the data cross-covariance terms from the distribution.
            preds = model.likelihood(model(test_x)).to_data_independent_dist()
            #print("Output of the Exact GP model is :"+str(preds.shape()))

        return preds.mean.mean(0), preds.variance.mean(0)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def make_orthogonal_vs(model, train_x):
    mean_inducing_points = torch.randn(1000, train_x.size(-1), dtype=train_x.dtype, device=train_x.device)
    covar_inducing_points = torch.randn(100, train_x.size(-1), dtype=train_x.dtype, device=train_x.device)

    covar_variational_strategy = gpytorch.variational.VariationalStrategy(
        model, covar_inducing_points,
        gpytorch.variational.CholeskyVariationalDistribution(covar_inducing_points.size(-2)),
        learn_inducing_locations=True
    )

    variational_strategy = gpytorch.variational.OrthogonallyDecoupledVariationalStrategy(
        covar_variational_strategy, mean_inducing_points,
        gpytorch.variational.DeltaVariationalDistribution(mean_inducing_points.size(-2)),
    )
    return variational_strategy

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from torch.utils.data import TensorDataset, DataLoader
# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)
# this is for running the notebook in our testing framework
#num_epochs = 1 if smoke_test else 1500


# Our testing script takes in a GPyTorch MLL (objective function) class
# and then trains/tests an approximate GP with it on the supplied dataset

def train_and_test_approximate_gp(model_cls, folderpath, train_x, train_y_N,y_trueSD, y_trueMean,test_x, test_y_N, n_iters):
    # DataLoader is a utility tool in the torch package used
    # to handle minibatches
    num_epochs = 1 if smoke_test else n_iters
    train_x = torch.squeeze(train_x)
    train_y_N =  torch.squeeze(train_y_N)
    
    test_x = torch.squeeze(test_x)
    test_y_N = torch.squeeze(test_y_N)
    train_dataset = TensorDataset(train_x, train_y_N)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        
    test_dataset = TensorDataset(test_x, test_y_N)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    inducing_points = torch.randn(12, train_x.size(-1), dtype=train_x.dtype, device=train_x.device)
    model = model_cls(inducing_points, train_x)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y_N.numel())
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.001)

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    # Training
    model.train()
    likelihood.train()
    epochs_iter = tqdm_notebook(range(num_epochs), desc=f"Training {model_cls.__name__}")
    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            #print(x_batch[:20,:])
            output = model(x_batch)
            #print("Before loss, the shapes,:")
            #print(output.shape,y_batch.shape)
            #print("trained output size is :"+str(output.shape), " Target output batch size is :"+str(y_batch.shape))
            #print(output[:20])
            loss = -mll(output, y_batch)
                        #print(loss.shape)
            epochs_iter.set_postfix(loss=loss.item())
            print('Iter %d - Loss: %.3f' % (i + 1, loss.item()))
            loss.backward()
            optimizer.step()

    
    # Testing
    model.eval()
    likelihood.eval()
    means = torch.tensor([0.])
    var = torch.tensor([0.])
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            preds = model(x_batch)
            #print(means.shape, preds.mean.cpu().shape)
            means = torch.cat([means, preds.mean.cpu()])
            var = torch.cat([var, preds.variance.cpu()])
    #means = (means[1:]*(y_max - y_min))+y_min
    #var = (var[1:]*(y_max - y_min))+y_min
    #means = means[1:]*(y_trueSD)+y_trueMean
    #var = var[1:]*(y_trueSD)+y_trueMean
    means = (means*(y_trueSD))+y_trueMean
    var = (var*(y_trueSD))+y_trueMean
    test_y = (test_y_N*(y_trueSD))+y_trueMean
    #means = torch.exp(means[1:])-1
    #var = torch.exp(var[1:])-1
    means = means[1:]
    var = var[1:]
    print(means.shape, var.shape)
    GPytorch_SVGP_2Norm = torch.linalg.norm(means - test_y.cpu())
    print(f"Test {model_cls.__name__} L2Norm: {GPytorch_SVGP_2Norm.item()}")
    if mssparkutils.fs.exists(folderpath):
        with open(folderpath+"NTX_Wthr_Decoupled_model.pkl", wb) as f:
            pickle.dump(model, f)
    """
    try:
        mssparkutils.fs.ls(folderpath)
        print("True")
    except:
        print("False")
    #with builtins.open("abfss://PC_Model@onelake.dfs.fabric.microsoft.com/PC_Model_Lakehouse.Lakehouse/Files/Models/NTX_Wthr_Decoupled_model.dump",'wb') as f:
    with builtins.open("Files/Models/NTX_Wthr_Decoupled_model.dump", "wb") as f:
        try:
            mssparkutils.fs.ls(modelpath)
            print("True")
        except:
            print("False")
        try:
            mssparkutils.fs.ls(folderpath)
            print("True")
        except:
            print("False")
        joblib.dump(model,f)  
    #torch.save(model.state_dict(), path)
    """
    return (means, var,GPytorch_SVGP_2Norm)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def test_approximate_gp(model_cls,PATH, test_x,y_trueSD, y_trueMean):
    test_dataset = TensorDataset(test_x)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    inducing_points = torch.randn(128, test_x.size(-1), dtype=test_x.dtype, device=test_x.device)
    model = model_cls(inducing_points, test_x)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # Loading and testing
    """
    model.load_state_dict(torch.load(PATH))
    """
    modelpath = str(PATH) +"NTX_Wthr_Decoupled_model.dump"
    with open(modelpath,"rb") as input_file:
        model = joblib.load(input_file)
    model.eval()
    
    likelihood.eval()
    
    means = torch.tensor([0.])
    var = torch.tensor([0.])
    with torch.no_grad():
        for x_batch in test_loader:
            x_batch = torch.stack(x_batch)
            preds = model(x_batch)
            model_mean = preds.mean.cpu()
            model_mean = torch.flatten(model_mean)
            model_var = preds.variance.cpu()
            model_var = torch.flatten(model_var)
            means = torch.cat([means,model_mean])
            var = torch.cat([var, model_var])
 
    means = means[1:]*(y_trueSD)+y_trueMean
    var = var[1:]*(y_trueSD)+y_trueMean

    print(f"Test {model_cls.__name__}")
    return (means, var)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## TRAINING MODEL

def run_model(model, train_x, train_y, y_true,  N, path, largeData):
    
    test_x = train_x
    test_y = train_y
    
    num_epochs = 150
    if largeData == 1: batch=128
    elif largeData ==0: batch = 5
    print("Inside run model, target output SD: "+str(y_trueSD)+", target output mean: "+str(y_trueMean))
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=train_y.size(0)))
    epochs_iter = tqdm_notebook(range(num_epochs), desc=f"Trainingthe exact GP model")
    for i in epochs_iter:
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            #hidden = repackage_hidden(hidden_layer)
            #print("trained output size is :"+str(output.shape), " Target output batch size is :"+str(y_batch.shape))
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
    model.eval()
    means = torch.tensor([0.])
    means = means.unsqueeze(0)
    var = torch.tensor([0.])
    var = var.unsqueeze(0)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():

        for x_batch, y_batch in test_loader:
            mean, variance= model.predict(model,x_batch)
            
            #print("for load "+str(load)+" the shape of mean is :"+str(mean.shape))
            means = torch.cat((means, mean))
                #if (tuple_count ==2):
            var = torch.cat((var, variance))
            #print("The size of the concatenated mean is :"+str(means.shape))
            #print("The size of the concatenated var is :"+str(var.shape))

    means = means[1:]
    var  = var[1:]
    #print(" the shape of final mean is :"+str(means.shape))
    #print(" the shape of final var is :"+str(var.shape))
   
    mean_np = means.numpy()
    var_np = var.numpy()
    
    
    arr = np.zeros(N)
    for i in range(N):
        arr[i]=(y_true[i])-(mean_np[i])
    
    clmdiff_norm_tr_test = norm(arr,2)

    
    #print("Saving to path: "+str(path))
    with open(os.path.join(path,"NTXWeather_DeepGP_model.dump"), wb) as f:
        joblib.dump(model,f)
    return means, var, clmdiff_norm_tr_test

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def test_model(model,path, XTest, YTest,  NTest, YTest_raw):
    test_dataset = TensorDataset(XTest, YTest)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)
    
    print("Loading from path: "+str(path))
    state_dict = torch.load(path)

    model.load_state_dict(state_dict, strict=False)
    #model = joblib.load(
    means = torch.tensor([0.])
    means = means.unsqueeze(0)
    var = torch.tensor([0.])
    var = var.unsqueeze(0)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for x_batch, y_batch in test_loader:
            mean, variance= model.predict(model,x_batch)
            
            #print("for load "+str(load)+" the shape of mean is :"+str(mean.shape))
            means = torch.cat((means_2021, mean))
                #if (tuple_count ==2):
            var = torch.cat((var_2021, variance))

    means = means[1:]
    var = var[1:]
   
    mean_np = means.numpy()
    var_np = var.numpy()
    
    
    arr = np.zeros(NTest)
    for i in range(NTest):
        arr[i]=(YTest_raw[i])-(mean_np[i])
    clmdiff_norm = norm(arr,2)
    return means, var, clmdiff_norm

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#abfss://1eae7a41-187c-448d-b33a-39a7f731ab8f@onelake.dfs.fabric.microsoft.com/16554174-10d3-4fb6-9c88-76c2efb59b51/Files/PreppedData
TrainingData = pd.read_csv("abfss://PC_Model@onelake.dfs.fabric.microsoft.com/PC_Model_Lakehouse.Lakehouse/Files/TrainData.csv")
ValidationData = pd.read_csv("abfss://PC_Model@onelake.dfs.fabric.microsoft.com/PC_Model_Lakehouse.Lakehouse/Files/ValData.csv")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import notebookutils
#abfss://PC_Model@onelake.dfs.fabric.microsoft.com/PC_Model_Lakehouse.Lakehouse/Files/Models
TrainedModels = notebookutils.fs.ls("abfss://PC_Model@onelake.dfs.fabric.microsoft.com/PC_Model_Lakehouse.Lakehouse/Files/Models")

if not TrainedModels:
    Train=1
else:
    for model in TrainedModels:
        print(model)
        if model.name == "NTX_Wthr_Decoupled_model.dump":
            Train=0
        else:
            print("Decoupled model not available, so training needs to be done")
            Train=1
"""
if notebookutils.fs.exists("abfss://PC_Model@onelake.dfs.fabric.microsoft.com/PC_Model_Lakehouse.Lakehouse/Files/Models/NTX_Wthr_Decoupled_model.dump"):
    Train=0
else:
    Train=1
"""
print(Train)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

Predictions_FilePath = "abfss://PC_Model@onelake.dfs.fabric.microsoft.com/PC_Model_Lakehouse.Lakehouse/Files/ModelPredictions/"
ModelsPath = "abfss://PC_Model@onelake.dfs.fabric.microsoft.com/PC_Model_Lakehouse.Lakehouse/Files/Models/"
try:
    mssparkutils.fs.ls("abfss://PC_Model@onelake.dfs.fabric.microsoft.com/PC_Model_Lakehouse.Lakehouse/Files/Models")
    print("True")
except:
    print("False")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def verify_path_exists(path):
    try:
        mssparkutils.fs.ls(path)
        return True
    except Exception:
        return False

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

verify_path_exists(ModelsPath)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

%run DataPrep_stage2_PostModelRun

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def train_predict_MultitaskDeepGP():
    if Train==1:
        ModelInput=TrainingData
        Phase = "Train and Test phase"
        yr = "2021 - 2022"
    else:
        ModelInput=ValidationData
        Phase = "Validation"
        yr= "2023"
        
    N = len(ModelInput)
    
    print(Phase+": Number of data: "+str(N))
    PreModelDataPrep = Dataprep_preModelInput(ModelInput)

    trtest_x, trtest_y, y_Mean, y_SD, N = PreModelDataPrep.DataProcess()
    model_GPytorch = MultitaskDeepGP(trtest_x.shape, trtest_y.shape, num_tasks=trtest_y.ndim)
    model_GPytorch = model_GPytorch.float()
    num_epochs= 400
    y = np.vstack(ModelInput['PollenCnt']).astype(int)
    if Train==1:
        model_mean, model_var, norm = run_model(model_GPytorch,trtest_x,trtest_y, y, y_SD,y_Mean, N, ModelsPath)
    else:
        model_mean, model_var,norm = test_model(model_GPytorch,ModelsPath,trtest_x,trtest_y, y_SD,y_Mean, N, y)
        
    #true_output = np.vstack(ModelInput['PollenCnt']).astype(int)
    PostModel_DataPrep = PostModelrun_DataPrep(ModelInput, model_mean, model_var, norm)
    PredData = PostModel_DataPrep.PredDataProcess()
    mean_np, var_np = PostModel_DataPrep.numpy_outputs_plots(num_epochs, Predictions_FilePath)
    
    cols=list(PredData.columns.values)
    print(cols)
    PredData.to_csv(os.path.join(Predictions_FilePath, "PollenCntActualPrediction_MultitaskDeepGaussianModel.csv"), index=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def train_predict_DSPP():
    if Train==1:
        ModelInput=TrainingData
        Phase = "Train and Test phase"
        yr = "2021 - 2022"
    else:
        ModelInput=ValidationData
        Phase = "Validation"
        yr= "2023"
        
    N = len(ModelInput)
    
    print(Phase+": Number of data: "+str(N))
    PreModelDataPrep = Dataprep_preModelInput(ModelInput)

    trtest_x, trtest_y, y_Mean, y_SD, N = PreModelDataPrep.DataProcess()
    num_epochs= 400
    if Train==1:
        model_mean, model_var, norm = train_and_test_DSPP(ModelsPath, trtest_x,trtest_y,trtest_x, trtest_y, N, num_epochs, y_Mean, y_SD)
    else:
        model_mean, model_var,norm = test_DSPP(ModelsPath,trtest_x,trtest_y, y_Mean, y_SD)

    PostModel_DataPrep = PostModelrun_DataPrep(ModelInput, model_mean, model_var, norm)
    PredData = PostModel_DataPrep.PredDataProcess()
    mean_np, var_np = PostModel_DataPrep.numpy_outputs_plots(num_epochs, Predictions_FilePath)
    
    PredData.to_csv(os.path.join(Predictions_FilePath, "PollenCntActualPrediction_DSPPGaussianModel.csv"), index=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def train_predict_Orthodecoupled():
    if Train==1:
        ModelInput=TrainingData
        Phase = "Train and Test phase"
        yr = "2021 - 2022"
    else:
        ModelInput=ValidationData
        Phase = "Validation"
        yr= "2023"
        
    N = len(ModelInput)
    num_tasks =1
    print(Phase+": Number of data: "+str(N))
    PreModelDataPrep = Dataprep_preModelInput(ModelInput)
    trtest_x, trtest_y, y_Mean, y_SD, N = PreModelDataPrep.DataProcess()
    model_GPytorch = MultitaskDeepGP(trtest_x.shape, trtest_y.shape, num_tasks=trtest_y.ndim)
    model_GPytorch = model_GPytorch.float()
    num_epochs= 1000
    y = np.vstack(ModelInput['PollenCnt']).astype(int)
    if Train==1:
        model_mean, model_var, norm = train_and_test_approximate_gp(OrthDecoupledApproximateGP, ModelsPath, trtest_x,trtest_y, y_SD, y_Mean,trtest_x,trtest_y,num_epochs)
        #:run_model(model_GPytorch,trtest_x,trtest_y, y, y_SD,y_Mean, N, PATH+PATHDECOUPLED)
    else:
        model_mean, model_var = test_approximate_gp(OrthDecoupledApproximateGP,ModelsPath, trtest_x, y_SD,y_Mean)
        #test_model(model_GPytorch,PATH+PATHDECOUPLED,trtest_x,trtest_y, y_SD,y_Mean, N, y, y_Mean)
        norm = torch.linalg.norm(model_mean - trtest_y)
    true_output = np.vstack(ModelInput['PollenCnt']).astype(int)
    
    PostModel_DataPrep = PostModelrun_DataPrep(ModelInput, model_mean, model_var, norm)
    PredData = PostModel_DataPrep.PredDataProcess()
    mean_np, var_np = PostModel_DataPrep.numpy_outputs_plots(num_epochs, Predictions_FilePath)
    
    PredData.to_csv(os.path.join(Predictions_FilePath, "PollenCntActualPrediction_OrthodecoupledGaussianModel.csv"), index=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

train_predict_Orthodecoupled()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
