# Zachary Cosenza
# Code for Desirability Bayesian Optimization

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Import Packages
from copy import deepcopy
import numpy as np
import torch
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim.optimize import optimize_acqf
from itertools import combinations
import gpytorch
from torch import Tensor
from gpytorch import constraints
from typing import Optional
from botorch.models.gp_regression import FixedNoiseGP
from botorch.acquisition.objective import MCAcquisitionObjective
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement

class DesirabilityFunctionObjective(MCAcquisitionObjective):

    def __init__(self,args) -> None:
        """
        Args:
            samples: A `sample_shape x batch_shape x q x m`-dim Tensors of samples from
                a model posterior.
            X: A `batch_shape x q x d`-dim Tensors of inputs.

        Returns:
            A `sample_shape x batch_shape x q x m'`-dim Tensor of objective values with
            `m'` the output dimension. This assumes maximization in each output
            dimension).
        """
        super().__init__()
        self.args = args

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        """
        This function gives desirability of a given set of inputs X based on the
        work of ...
        
        B. Akteke-Ozturk, G. Koksal, and G. W. Weber, “Nonconvex optimization of 
        desirability functions,” Qual. Eng., vol. 30, no. 2, pp. 293–310, 2018, 
        doi: 10.1080/08982112.2017.1315136.
        
        D(x) = sqrt(x_bar * y_bar + beta)
        beta = 0.1, which improves numerical stability
        y_bar = ((y - 0.5) / (2 - 0.5)).pow(1), which encorporates the sample quality
        c_bar = ((cost - 0.020607015 + 19.23264982) / 
                 (0.020607015 - 0.020607015 + 19.23264982)).pow(1)
                , which encorporates the sample cost
        
        Args:
            X: batch_shape x n x px+1, which is input of optimization problem
            samples : mcmc_shape x batch_shape x n x 1, output samples from sampler
            common_mean: scalar, which is mean of raw data
            common_std: scalar, which is std of raw data
        Returns:
            D : batch_shape x n, which is desirability of input X
        """
        #cost : batch_shape x n
        
        #Cost (Minimize)
        
        #Customize this ----------------------------------------------------
        c = torch.from_numpy(np.ones(X.shape[-1]-1)).to(dtype = torch.float32)
        cost = torch.matmul(X[...,:-1],c)
        cost_min = 0
        cost_max = cost_min + (X.shape[-1] - 1)
        # ------------------------------------------------------------------
        
        cost_s = 1 #degree of importance (set at 1 as default)
        
        #Normalize Cost
        d_cost = ((cost - cost_max) / (cost_min - cost_max)).pow(cost_s)
        if d_cost.dim() == 1:
            d_cost = d_cost.view(samples.shape[1],len(d_cost))
            
        #Restandardize f(x)
        # common_mean,common_std = self.args[0],self.args[1]
        # samples = samples * common_std + common_mean
        
        #No Restandardization of f(x)
        common_mean,common_std = self.args[0],self.args[1]
        # samples = samples * common_std + common_mean
        f_min = -2
        f_max = 2
        
        #f(x) (Maximize)
        # f_min = 0.5
        # f_max = 2
        f_s = 1 #degree of importance (set at 1 as default)
        mask = samples < f_min
        d_f = ((samples - f_min) / (f_max - f_min)).pow(f_s)
        d_0 = torch.zeros(1).expand_as(d_f)
        d_f[mask] = d_0[mask]
        
        #Desirability Function
        equation = 'kbij,bi->kbij'
        D = (torch.einsum(equation,d_f,d_cost) + 0.1) **(0.5)
        
        return D.squeeze(-1)

class MultiIS_ARD_Kernel(gpytorch.kernels.Kernel):
    
    """Multi-Information Source ARD Kernel

    This Kernel is based on the kernel developed in ...
    
    M. Poloczek, J. Wang, and P. I. Frazier, “Multi-information 
    source optimization,” 2017.
    
    Args:
        px: scalar, number of dimensions in X
        num_IS: scalar, number of information sources
        x1,x2: batch_shape x n x px + 1: inputs
    Returns:
        K: n x n, kernel evaluations K
    """
    
    def __init__(self,num_IS,num_dim):
        super(MultiIS_ARD_Kernel, self).__init__()
        self.num_IS = num_IS
        self.num_dim = num_dim
        self.register_parameter(name="l", parameter=torch.nn.Parameter(torch.ones(num_IS*num_dim)))
        self.register_parameter(name="s", parameter=torch.nn.Parameter(torch.ones(num_IS)))
        l_prior = gpytorch.priors.MultivariateNormalPrior(torch.ones(num_IS*num_dim),torch.eye(num_IS*num_dim)/2**2)
        s_prior = gpytorch.priors.MultivariateNormalPrior(torch.ones(num_IS),torch.eye(num_IS)/2**2)
        self.register_prior("l_prior",l_prior,"l")
        self.register_prior("s_prior",s_prior,"s")
        self.register_constraint(param_name = "l",
                                  constraint = constraints.Interval(0.001, 100))
        self.register_constraint(param_name = "s",
                                  constraint = constraints.Interval(0.001, 100))
    def forward(self,x1,x2,diag=False,last_dim_is_batch=False):
        num_dim = self.num_dim
        num_IS = self.num_IS
        IS = x1[...,-1].long().unsqueeze(-1)
        L0 = self.l[0:num_dim]
        x1_0 = x1[...,:-1].contiguous().div(L0)
        x2_0 = x2[...,:-1].contiguous().div(L0)
        d0 = torch.cdist(x1_0,x2_0).pow(2)
        K = self.s[0].pow(2) * torch.exp(-d0 / 2)
        for j in np.arange(num_IS-1):
            mask = (IS == j+1) * (IS == x2[...,-1].unsqueeze(-2))
            L = self.l[(j+1)*num_dim:(j+1)*num_dim+num_dim]
            x1_j = x1[...,:-1].contiguous().div(L)
            x2_j = x2[...,:-1].contiguous().div(L)
            d_j = torch.cdist(x1_j,x2_j).pow(2)
            s = self.s[j+1].expand_as(K)
            K[mask] = K[mask] + s[mask].pow(2) * torch.exp(-d_j[mask] / 2)
        return K

# Import or Generate Data
N = 200
X = np.random.uniform(0,1,size = (N,19)) #this must be n_datapoints x dimension
X[:,-1] = np.random.choice(np.arange(3),size = N) #last dimension must be indicator (see below)
n_replicates = 6
Y = np.repeat(- np.sum((X[:,:-1] - 0.5)**2,axis = 1),n_replicates).reshape((N,n_replicates)) #output n_datapoints x n_replicates
#indicator: last column of X, 0 = true data, 1,2,3,4... = approximations of true data

#Standardize Data with all Replicates
Y_standardized = (Y - Y.mean()) / Y.std()
Y_var_standardized = np.var(Y, axis = 1)
Y_mean_standardized = Y.mean(axis = 1)

#Initalize Training and Testing Data
num_IS = 4
num_dim = X.shape[1] - 1 #not including fidelity
HIGH_FIDELITY_BATCH_SIZE = 3
LOW_FIDELITY_BATCH_SIZE = 5
BATCH_SIZE = HIGH_FIDELITY_BATCH_SIZE + LOW_FIDELITY_BATCH_SIZE

MC_SAMPLES = 2000
NUM_RESTARTS = 5
RAW_SAMPLES = 1024

# Reformat Data
X_ = torch.from_numpy(X).to(dtype = torch.float32).view((1,X.shape[0],X.shape[1])) #reshape data
Y_ = torch.from_numpy(Y_mean_standardized).to(dtype = torch.float32).view((1,len(Y),1)) #reshape data
Y_var_= torch.from_numpy(Y_var_standardized).to(dtype = torch.float32).view((len(Y),1)) #reshape data

# Set MC Sampler
qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

# Set Covariance Type
covar_module = MultiIS_ARD_Kernel(num_IS,num_dim)

# Set Model Type
model = FixedNoiseGP(train_X = X_, 
                      train_Y = Y_, 
                      train_Yvar = Y_var_,
                      covar_module = covar_module)

# Set Prior on Homoscedastic Noise Model
noise_prior = gpytorch.priors.GammaPrior(concentration = 1.1, rate = 0.05)
noise_constraint = constraints.Positive()

# Make Model Utilize Measured Y_var_ (Variance) = Heteroscedastic Noise Model
model.likelihood = FixedNoiseGaussianLikelihood(noise = Y_var_.flatten(), 
                                                    learn_additional_noise=True,
                                                    noise_prior=noise_prior,
                                                    noise_constraint=noise_constraint)

# Optimize Hyperparameters
mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_model(mll)

#Set Up Monte Carlo Sampler
bounds = torch.tensor([[0.0] * (num_dim+1), [1.0] * (num_dim+1)]) #bounds should be [0,1]
bounds[-1,-1] = 1.0 * (num_IS - 1) #last bound should allow for fidelity indicator, not needed usually
qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

# Set Up Optimization Objective
objective = DesirabilityFunctionObjective([Y.mean(),Y.std()])

aq_funct = qNoisyExpectedImprovement(model = model,
                                     X_baseline = X_[X_[...,-1]==0,:], #baseline should be only highest fidelity experiments
                                     sampler = qmc_sampler,
                                     objective = objective,
                                     )

# Optimize Problem
fixed_features_dict = {num_dim: 0.0}
x_opt , y_opt = optimize_acqf(acq_function=aq_funct,
                                bounds=bounds,
                                q=BATCH_SIZE,
                                num_restarts=NUM_RESTARTS,
                                raw_samples=RAW_SAMPLES,
                                fixed_features=fixed_features_dict)

#Apply Choice Rule
combo_list = list(combinations(np.arange(BATCH_SIZE),HIGH_FIDELITY_BATCH_SIZE))
results = np.zeros(len(combo_list))
for i in np.arange(len(combo_list)):
    results[i] = aq_funct(x_opt[combo_list[i],:])
combo_opt = combo_list[np.argmax(results)]

#Attach Conditions to Optimal X and Y
X_highfid = deepcopy(x_opt[combo_opt,:])
X_lowfid = deepcopy(x_opt)

#Print Results
print(X_highfid,X_lowfid)