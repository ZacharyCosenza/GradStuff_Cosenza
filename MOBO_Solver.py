# Zachary Cosenza
# MultiObjective Bayesian Optimization (MOBO)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Import Packages
import numpy as np
import torch
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim.optimize import optimize_acqf
import gpytorch
from torch import Tensor
from gpytorch import constraints
from botorch.fit import fit_gpytorch_model
from typing import Optional
from botorch.models.gp_regression import FixedNoiseGP
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from itertools import combinations
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

class MultiIS_ARD_Kernel(gpytorch.kernels.Kernel):
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

class StochasticDeterministicMultiObjective(MCMultiOutputObjective):

    def __init__(self,DeterministicFunct) -> None:
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
        self.DeterministicFunct = DeterministicFunct

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        
        # Deterministic Function 1 (Must Follow Shape of samples)
        # (Maximize This So Make Negative)
        DeterministicFunct = self.DeterministicFunct
        f_1 = - DeterministicFunct(X,samples)
        
        # Stochastic Function 2 (Maximize This)
        f_2 = samples
        
        # Combine Functions for MultiObjective Case
        f_moo = torch.cat((f_2, f_1), axis = -1)
        
        return f_moo

def MediaCostFunction(x,samples):
    num_dim = x.shape[-1] - 1 #account for IS
    c = torch.tensor(np.array([0, #MEM-NEAA
                        0,#MEM-EAA
                        0,#MEM-Vitamin
                        0,#Salts
                        0,#Trace Metals
                        0,#DNA Precursor
                        0,#Fatty Acids
                        0,#Sodium Selenite
                        0,#Ascorbic Acid
                        0,#Glucose
                        0,#Glutamine
                        0,#Sodium Pyruvate
                        0,#Sodium Chloride
                        0.030447656,#Insulin
                        0.003889299,#Transferrin
                        0.633400136,#FGF2
                        0.088898265,#TGFb1
                        0.003583711,#EGF
                        1.69462E-08,#Progesterone
                        7.17553E-08,#Estradiol
                        0.083342123,#IL6
                        0.015557196,#LIF
                        0.036981678,#TGFb3
                        0.033336849,#HGF
                        0.027225094,#PDGF
                        0.043337904,#PEDF
                        ])).reshape((num_dim,1)).to(dtype = torch.float32)

    if samples.dim() == 3:
        X_ = x.unsqueeze(0).repeat(samples.shape[0],1,1)
    else:
        X_ = x.unsqueeze(0).repeat(samples.shape[0],1,1,1)
    f = X_[...,:-1] @ c
    return f

def Compute_Cost_Media(x):
    c = np.array([0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0.030447656,
                        0.003889299,
                        0.633400136,
                        0.088898265,
                        0.003583711,
                        1.69462E-08,
                        7.17553E-08,
                        0.083342123,
                        0.015557196,
                        0.036981678,
                        0.033336849,
                        0.027225094,
                        0.043337904])
    return x[...,:-1] @ c
     
def Feasibility_Constraint_Callable(samples,args):
    '''
    A list of callables, each mapping a Tensor of dimension
    `sample_shape x batch-shape x q x m` to a Tensor of dimension
    `sample_shape x batch-shape x q`, where negative values imply
    feasibility. The acqusition function will compute expected feasible
    hypervolume.
    WANT: Posterior(x) >= c
          Posterior(x) - c >= 0
    '''
    mu0 = args[0]
    std0 = args[1]
    c = -3 #unstandardized constant to get greater than
    con = samples - (c - mu0) / std0
    return - con.squeeze(-1) # negative = feasible

#Collect X and Y and Y_var (numpy arrays)
X = np.random.uniform(0,1,(100,26+1))
X[:,-1] = 0
X[:50,-1] = 1
Y = -np.sum((X[:,:-1] - 0.5) ** (2),axis = 1)

#Standardize for GP Model
Y_standardized = (Y - Y.mean()) / Y.std()
Y_var_standardized = np.abs(Y_standardized / 4)
C = Compute_Cost_Media(X)

#Convert X and Y and Y_var to Torch Tensors in Proper Format
Y_ = torch.from_numpy(Y_standardized).view((len(Y),1)).to(dtype = torch.float32)
X_ = torch.from_numpy(X).to(dtype = torch.float32)
Y_var_ = torch.from_numpy(Y_var_standardized).view((len(Y),1)).to(dtype = torch.float32)
C_ = torch.from_numpy(C.reshape(len(Y),1))

#Initalize Training and Testing Data
num_IS = 4
num_dim = X.shape[1] - 1 #not including fidelity
BATCH_SIZE = 10
HIGH_FIDELITY_BATCH_SIZE = 3

MC_SAMPLES = 2000
NUM_RESTARTS = 5
RAW_SAMPLES = 1024
bounds = torch.tensor([[0.0] * (num_dim+1), [1.0] * (num_dim+1)])
qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

#Ref Point at Worst Point Possible/Reasonable
Y_ref = -4 #4 standard deviations below mean value
C_ref = -1.1 #higher than highest cost possible (-1)
reference_point = torch.tensor((Y_ref,C_ref))
                
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
    
# Make Model Utilize Measured Y_var (Variance) = Heteroscedastic Noise Model
model.likelihood = FixedNoiseGaussianLikelihood(noise = Y_var_.flatten(), 
                                                    learn_additional_noise=True,
                                                    noise_prior=noise_prior,
                                                    noise_constraint=noise_constraint)
    
# Optimize Hyperparameters
mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_model(mll)
    
# Define Objective and A Function
objective = StochasticDeterministicMultiObjective(MediaCostFunction)
constraint_list = [lambda samples: Feasibility_Constraint_Callable(samples, 
                                                                   args = [Y.mean(),Y.std()])]
aq_funct = qNoisyExpectedHypervolumeImprovement(
    model=model,
    ref_point=reference_point, 
    X_baseline=X_[X[:,-1]==0,:],
    sampler=qmc_sampler,
    prune_baseline=True,
    objective = objective,
    constraints = constraint_list,
)
    
# Optimize Problem
fixed_features_dict = {num_dim: 0.0}
x_opt, y_opt = optimize_acqf(
    acq_function=aq_funct,
    bounds=bounds,
    q = BATCH_SIZE,
    num_restarts=NUM_RESTARTS,
    raw_samples=RAW_SAMPLES,  # used for intialization heuristic
    sequential=True,
    fixed_features = fixed_features_dict,
)
    
#Apply Choice Rule (Combine Combos with All Data per Combo)
combo_list = list(combinations(np.arange(BATCH_SIZE),HIGH_FIDELITY_BATCH_SIZE))
results = np.zeros(len(combo_list))
current_data = torch.cat((Y_,-C_),axis = -1)
for i in np.arange(len(combo_list)):
    with torch.no_grad():
        temp_y = model(x_opt[combo_list[i],:]).mean
    temp_c = - Compute_Cost_Media(x_opt[combo_list[i],:])
    temp_data = torch.cat((temp_y.reshape((HIGH_FIDELITY_BATCH_SIZE,1)),
                           temp_c.reshape((HIGH_FIDELITY_BATCH_SIZE,1))),axis = -1)
    temp_total_data = torch.cat((current_data,temp_data))
    vol = DominatedPartitioning(ref_point = reference_point,
                        Y = temp_data).compute_hypervolume().item()
    results[i] = vol
combo_opt = combo_list[np.argmax(results)]

print(x_opt,y_opt)
print(x_opt[combo_opt,:])

#%%Plot Resulting Datapoints----------------------------------------------------

import matplotlib.pyplot as plt

plt.figure()
n = 20
for i in np.arange(num_dim):
    x = torch.ones((n,num_dim+1))
    x[:,-1] = 0
    x[:,i] = torch.linspace(0,1,n)
    with torch.no_grad():
        y = model.posterior(x).mean.numpy()
    plt.subplot(5,6,i+1)
    plt.plot(np.linspace(0,1,n),y,'k')
    plt.vlines(x_opt[:,i],y.min(),y.max(),colors = 'b')
    plt.vlines(x_opt[combo_opt,i],y.min(),y.max(),colors = 'r')
    plt.xticks([])
    plt.yticks([])

Y_opt = -np.sum((x_opt.numpy()[:,:-1] - 0.5) ** (2),axis = 1)
Y_opt0 = -np.sum((x_opt.numpy()[combo_opt,:-1] - 0.5) ** (2),axis = 1)
C_opt = Compute_Cost_Media(x_opt.numpy())
C_opt0 = Compute_Cost_Media(x_opt[combo_opt,:].numpy())

with torch.no_grad():
    y_test_opt = model.posterior(x_opt).mean.numpy()
    y_test_opt0 = model.posterior(x_opt[combo_opt,:]).mean.numpy()

plt.figure()
plt.plot(C,Y_standardized,'ks')
plt.plot(C_opt,(Y_opt - Y.mean()) / Y.std(),'bs')
plt.plot(C_opt0,(Y_opt0 - Y.mean()) / Y.std(),'rs')
plt.plot(C_opt,y_test_opt,'bx')
plt.plot(C_opt0,y_test_opt0,'rx')
plt.xlabel('cost $c(x)$')
plt.ylabel('growth $y(x)$')

#-------------------------------------------------------------------------------

#%% Initalization using Latin Hypercube Sample----------------------------------

import pyDOE as doe

N_init = 20 #inital amount of samples
num_IS = 4 #number of information source to consider
num_dim = 26 #not including fidelity
num_highest_IS = 6

X_init = doe.lhs(num_dim, samples = N_init)

#These will be selected to be highest fidelity, other are lowest
IS_init = np.random.choice(np.arange(N_init),num_highest_IS,replace = False)

#Save to Excel-Readable Format

import pandas as pd

## convert your array into a dataframe
df = pd.DataFrame(X_init)
df_IS = pd.DataFrame(IS_init)

## save to xlsx file
filepath = 'X_init_MOBO.xlsx'
filepath_IS = 'IS_init_MOBO.xlsx'
# df.to_excel(filepath, index=False)
# df_IS.to_excel(filepath_IS, index=False)

#------------------------------------------------------------------------------