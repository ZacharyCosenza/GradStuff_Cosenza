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
from gpytorch import constraints
from botorch.fit import fit_gpytorch_model
from botorch.models.gp_regression import FixedNoiseGP
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from itertools import combinations
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
import pandas
from torchtoolbox import fit_gpytorch_multistart, MultiIS_ARD_Kernel, StochasticDeterministicMultiObjective, MediaCostFunction, Compute_Cost_Media
from torchtoolbox import Feasibility_Constraint_Callable

# Import Data and Convert to Numpy Array

file_name = 'C:/Users/zacco/Documents/Zac_ResearchFiles/Python_Research/Data/MOBO_Data/X.txt'
# file_name = 'C:/Users/Zachary Cosenza/Documents/Zac_Research/ResearchFiles/Python_Research/Data/MOBO_Data/X.txt'
X_df = df = pandas.read_csv(file_name, delimiter = '\t', header = None)

file_name = 'C:/Users/zacco/Documents/Zac_ResearchFiles/Python_Research/Data/MOBO_Data/Y.txt'
# file_name = 'C:/Users/Zachary Cosenza/Documents/Zac_Research/ResearchFiles/Python_Research/Data/MOBO_Data/Y.txt'
Y_df = df = pandas.read_csv(file_name, delimiter = '\t', header = None)

# Extract Mean and Standard Deviation from Data
common_mean = Y_df.stack().mean()
common_std = Y_df.stack().std()

# Standardize Data
Y_standardized_df = (Y_df - common_mean) / common_std

# Reduce Data to Mean and Variance and Convert to Numpy
min_var_standardized = 0.02
Y_standardized = Y_standardized_df.mean(axis = 1).to_numpy()
Y_var_standardized = Y_standardized_df.var(axis = 1).to_numpy()
X = X_df.to_numpy()

# Compute Cost
C = Compute_Cost_Media(X)

#Convert X and Y and Y_var to Torch Tensors in Proper Format
Y_ = torch.from_numpy(Y_standardized).view((len(Y_standardized),1)).to(dtype = torch.float32)
X_ = torch.from_numpy(X).to(dtype = torch.float32)
Y_var_ = torch.from_numpy(Y_var_standardized).view((len(Y_var_standardized),1)).to(dtype = torch.float32)
Y_var_ = torch.clamp(Y_var_,min = min_var_standardized)
C_ = torch.from_numpy(C.reshape(len(Y_var_standardized),1))

#Initalize Training and Testing Data
num_IS = 4
num_dim = X.shape[1] - 1 #not including fidelity
BATCH_SIZE = 15
HIGH_FIDELITY_BATCH_SIZE = 3

MC_SAMPLES = 2000
NUM_RESTARTS = 20
RAW_SAMPLES = 1024
bounds = torch.tensor([[0.0] * (num_dim+1), [1.0] * (num_dim+1)])
bounds[1][-1] = int(num_IS - 1)
bounds[0][:17] = 0.1
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
# fit_gpytorch_model(mll)
# mll, state_dict_list = fit_gpytorch_multistart(mll)

# Loading Model if Needed
model_path = 'C:/Users/zacco/Documents/Zac_ResearchFiles/Python_Research/Data/MOBO_Data/MOBO_Model13.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

#%% Optimization of Hypervolume Function
    
# Define Objective and A Function
objective = StochasticDeterministicMultiObjective(MediaCostFunction)
constraint_list = [lambda samples: Feasibility_Constraint_Callable(samples, 
                                                                    args = [common_mean,common_std])]
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

#%% Optimization of Posterior using NEI Function (No Constraints Needed)

from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement

aq_funct = qNoisyExpectedImprovement(
            model = model, 
            X_baseline=X_[X[:,-1]==0,:],
            sampler=qmc_sampler,
        )

# Optimize Problem
fixed_features_dict = {num_dim: 0.0}
x_opt, _ = optimize_acqf(
    acq_function=aq_funct,
    bounds=bounds,
    q = 1,
    num_restarts = NUM_RESTARTS,
    raw_samples = RAW_SAMPLES,  # used for intialization heuristic
    fixed_features = fixed_features_dict,
)
print(x_opt)

#%% Homotopy Optimization for Sparse-BO

from torchtoolbox import MediaComplexityFunction

#Ref Point at Worst Point Possible/Reasonable
Y_ref = -4 #4 standard deviations below mean value
n_ref = -num_dim #higher than highest dimensionality possible -num_dim
reference_point_complexity = torch.tensor((Y_ref,n_ref))

# Define Objective and A Function
objective = StochasticDeterministicMultiObjective(MediaComplexityFunction)
aq_funct = qNoisyExpectedHypervolumeImprovement(
    model=model,
    ref_point=reference_point_complexity, 
    X_baseline=X_[X[:,-1]==0,:],
    sampler=qmc_sampler,
    prune_baseline=True,
    objective = objective,
)

# Optimize Problem
fixed_features_dict = {num_dim: 0.0}
x_opt, _ = optimize_acqf(
    acq_function=aq_funct,
    bounds=bounds,
    q = 1,
    num_restarts = NUM_RESTARTS,
    raw_samples = RAW_SAMPLES,  # used for intialization heuristic
    fixed_features = fixed_features_dict,
)
print(x_opt)

#%%Plot Resulting Datapoints----------------------------------------------------

labels = ['NEAA','EAA','V','Salt','Metals','DNA',
          'Fat','SS','AA','Gluc','Glut','Pyruvate',
          'NaCl','I','T','FGF2','TGFb1','EGF','P',
          'Estra','IL6','LIF','TGFb3','HGF','PDGF','PEDF']

import matplotlib.pyplot as plt

plt.figure()
n = 20
for i in np.arange(num_dim):
    x = torch.ones((n,num_dim+1)) * 0.5
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
    # plt.hist(x_opt[:,i])
    plt.title(labels[i])

C_opt = Compute_Cost_Media(x_opt.numpy())
C_opt0 = Compute_Cost_Media(x_opt[combo_opt,:].numpy())

with torch.no_grad():
    y_test_opt = model.posterior(x_opt).mean.numpy()
    y_test_opt0 = model.posterior(x_opt[combo_opt,:]).mean.numpy()

plt.figure()
plt.plot(C,Y_standardized,'ks')
plt.plot(C_opt,y_test_opt,'bx')
plt.plot(C_opt0,y_test_opt0,'rx')
plt.xlabel('cost $c(x)$')
plt.ylabel('growth $y(x)$')

#Print Parameters
print(model.covar_module.l,
      model.covar_module.s,
      model.likelihood.second_noise_covar.noise)

#%% Initalization using Latin Hypercube Sample----------------------------------

# import pyDOE as doe
import numpy as np

# N_init = 30 #inital amount of samples
# num_IS = 4 #number of information source to consider
# num_dim = 17 #not including fidelity
# num_highest_IS = 3

# X_init = doe.lhs(num_dim, samples = N_init)

#These will be selected to be highest fidelity, other are lowest
# IS_init = np.random.choice(np.arange(N_init),num_highest_IS,replace = False)

#Save to Excel-Readable Format

import pandas as pd

## convert your array into a dataframe
df_2 = pd.DataFrame(x_opt.numpy())
df_1 = pd.DataFrame(x_opt[combo_opt,:].numpy())
# df_2 = pd.DataFrame(X_init)
# df_1 = pd.DataFrame(IS_init)

## save to xlsx file
filepath_1 = 'X_MOBO13_highfid.xlsx'
filepath_2 = 'X_MOBO13_lowfid.xlsx'
# df_1.to_excel(filepath_1, index=False)
# df_2.to_excel(filepath_2, index=False)

#------------------------------------------------------------------------------

#%% Save Model State

name_model = 'MOBO_Model13.pth'
# torch.save(model.state_dict(), name_model)