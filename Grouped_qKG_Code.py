# Zachary Cosenza
# Test of Infrastructure for Optimization Sequence KG

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
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from copy import deepcopy
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib.pyplot as plt
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
import torchtoolbox as ttb
from tqdm import tqdm
from botorch.models.gp_regression import FixedNoiseGP   

# NOTE: i used this area for messing around and plots

# Define Toy Problem
def toyproblem(x):
    return -np.sum((x[:,:-1] - 0.5) ** (2),axis = 1) + x[:,-1] * x[:,0]
    #Get Fidelities
    ind_0 = x[:,-1] == 0
    ind_1 = x[:,-1] == 1
    ind_2 = x[:,-1] == 2
    c = np.array([0,0.5,0.3,0.8,1])
    y = np.zeros(x.shape[0])
    d = np.array([1,2,-3])
    y[ind_0] = np.sum((x[ind_0,:5] - c)**2,axis = 1) + np.sum(x[ind_0,5:-1],axis = 1)
    y[ind_1] = np.sum((x[ind_1,:5] - c - 0.2)**2,axis = 1) + np.sum(x[ind_1,5:-1],axis = 1)
    y[ind_2] = np.sum((x[ind_2,:5] - c + 0.2)**2,axis = 1) + np.sum(x[ind_2,5:-1],axis = 1)
    y = y + d[0] * x[:,0] * x[:,1] + d[1] * x[:,5] * x[:,3] + d[2] * x[:,7] * x[:,8]
    y = y + np.random.uniform(-0.35,0.35,size = len(y))
    y[ind_1] = y[ind_1] - 2 * (x[ind_1,1] + 3 * x[ind_1,7] - x[ind_1,9]) + np.random.uniform(-0.35,0.35,size = len(y[ind_1]))
    y[ind_2] = 2 * y[ind_2] + 2 * (x[ind_2,0] - 3 * x[ind_2,4] + x[ind_2,8]) + 2 * x[ind_2,0] * x[ind_2,4] - 2 * x[ind_2,1] * x[ind_2,2]
    return y

#Parameters
num_IS = 2 #number of information sources
N = 100 #inital training datapoints
n = 100 #testing datapoints
num_dim = 1 #dimensionality of the toyproblem
num_samples = 2000  #mcmc samples

#Collect X and Y and Y_var (numpy arrays)
X = np.random.uniform(0,1,(N,num_dim+1))
X[:,-1] = 1
X[:int(N/5),-1] = 0
Y = toyproblem(X)

#Testing Dataset 0
x0 = np.random.uniform(0,1,(n,num_dim+1))
x0[:,-1] = 0
x0[:,0] = np.linspace(0,1,n)
y0 = toyproblem(x0)

#Testing Dataset 1
x1 = np.random.uniform(0,1,(n,num_dim+1))
x1[:,-1] = 1
x1[:,0] = np.linspace(0,1,n)
y1 = toyproblem(x1)

# Set Covariance Type as Multi-IS Kernel
covar_module = ttb.MultiIS_Kernel(num_IS)
    
# Set Model Type as SingleTaskGP
train_X = torch.from_numpy(X).to(dtype = torch.float32)
train_Y = torch.from_numpy(Y).view((len(Y),1)).to(dtype = torch.float32)
model = SingleTaskGP(train_X = train_X, 
              train_Y = train_Y, 
              covar_module = covar_module)
mll = ExactMarginalLogLikelihood(model.likelihood, model)

#Hyperparameter Training
fit_gpytorch_model(mll)

#Predict Dataset and Plot R2
with torch.no_grad(): 
    y_pred0 = model.posterior(torch.from_numpy(x0).to(dtype = torch.float32)).mean
    y_pred1 = model.posterior(torch.from_numpy(x1).to(dtype = torch.float32)).mean

#Set Sampler
sampler = SobolQMCNormalSampler(num_samples = num_samples)

#Set Posterior
test_x0 = torch.from_numpy(x0).to(dtype = torch.float32)
test_x1 = torch.from_numpy(x1).to(dtype = torch.float32)
posterior0 = model.posterior(test_x0)
posterior1 = model.posterior(test_x1)

#Sample from Posterior
samples0 = sampler(posterior0)
samples1 = sampler(posterior1)

plt.figure()
plt.errorbar(x0[:,0], 
              samples0.detach().numpy().mean(axis = 0), 
              yerr=samples0.detach().numpy().std(axis = 0).flatten(),
              color = 'r')
plt.errorbar(x1[:,0], 
              samples1.detach().numpy().mean(axis = 0), 
              yerr=samples1.detach().numpy().std(axis = 0).flatten(),
              color = 'b')
plt.plot(X[X[:,-1] == 0,0],Y[X[:,-1] == 0],'rs')
plt.plot(X[X[:,-1] == 1,0],Y[X[:,-1] == 1],'bs')

# Set Function to be Optimized
num_fantasies = 10
AcqusitionFunction = qKnowledgeGradient(
    model = model,
    num_fantasies = num_fantasies,
    current_value = train_Y[train_X[:,-1]==0].max(),
)

# Boundary
bounds = torch.tensor([[0.0] * (num_dim+1), [1.0] * (num_dim+1)])
bounds[-1][-1] = int(num_IS)

# Optimization Parameters
q = 10 #number of points to solve for
num_restarts = 5 #number of optimization restarts
raw_samples = 200

# Fixed Features List
fixed_features_list_IC = {num_dim: 0.0}
fixed_features_list = {num_dim: torch.ones(q+num_fantasies).view(q+num_fantasies,1)}
fixed_features_list[num_dim][round(q/2):] = 0.0

# Initial Conditions
batch_initial_conditions_tensor = gen_one_shot_kg_initial_conditions(
    acq_function=AcqusitionFunction,
    bounds=bounds,
    q=q,
    num_restarts=num_restarts,
    raw_samples=raw_samples,
    fixed_features=fixed_features_list_IC,
)

# Optimize Problem
candidates, acq_value = optimize_acqf(
    acq_function = AcqusitionFunction,
    bounds = bounds,
    q = q,
    num_restarts = num_restarts,
    raw_samples = raw_samples,
    fixed_features = fixed_features_list,
    maximize = True,
    batch_initial_conditions = batch_initial_conditions_tensor,
)

print(candidates,acq_value)

plt.vlines(candidates[candidates[:,-1]==0,0],0,1,'r')
plt.vlines(candidates[candidates[:,-1]==1,0],0,1,'b')
plt.xlabel('x')
plt.ylabel('y')

#%% Optimization Test for GroupedKG (this area for optimization loops)

# Define Test Problem for Optimization
toyproblem = ttb.get_bowlline_hard

# Name of Final File
file_name_output = ''

#Initalize Training and Testing Data
num_IS = 3 #number of information sources
num_fantasies = 16 #for qkg
N = 15 #total inital experiments via latin hypercube
N_0 = 2 #total inital 0th fidelity experiments
num_dim = 10 #not including fidelity
q_0 = 2 #number of 0th fidelity experiments / batch
q_real = 5 #total number of experiments / batch
q_1 = 2 #number of 1st fidelity experiments / batch
q_2 = q_real - q_1 #number of 2nd fidelity experiments / batch
num_restarts = 10 #number of optimization restarts
raw_samples = 1000

B = 10 #number of batches of experiments per iteration
num_blocks = B + 1
max_iters = 20 #number of iterations
Y_record = torch.zeros([max_iters,B+1])
n_reps = 6 #number of samples of each toy problem per evaluation (they have noise)

# Boundary
bounds = torch.tensor([[0.0] * (num_dim+1), [1.0] * (num_dim+1)])
bounds[-1][-1] = int(num_IS)

# Set Covariance Type as Multi-IS Kernel
covar_module = ttb.MultiIS_Kernel(num_IS)

# Set Sampler
MC_SAMPLES = 2000
qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

# Fixed Features List ---------------------------------------------------------
# this holds the constraints in the optimization problem
key = num_dim
fixed_features_list_IC = {key: 0.0}
fl_size = q_0 + q_1 + q_2 + num_fantasies
# Initalize w/ 0th Fidelity
fixed_features_list = {key: torch.zeros(fl_size).view(fl_size,1)}
# Fidelity 1
fixed_features_list[key][q_0:q_0 + q_1] = 1.0
# Fidelity 2
fixed_features_list[key][q_0 + q_1:q_0 + q_1 + q_2] = 2.0
# ----------------------------------------------------------------------------

for i in tqdm(np.arange(max_iters)):
    
    #Get Initial Experiments via DOE Latin Hypercube
    X,Y = ttb.get_training_data(N,N_0,num_dim,toyproblem,num_IS)
    Y_standardized = (Y - Y.mean()) / Y.std()
    Y_var_standardized = ttb.get_variance_standardized(X,Y,toyproblem,n_reps)
    Y_record[i,0] = Y[X[:,-1]==0].max()
    
    for b in np.arange(B):
        
        # Set Model Type
        train_X = torch.from_numpy(X).to(dtype = torch.float32)
        train_Y = torch.from_numpy(Y_standardized).view((len(Y),1)).to(dtype = torch.float32)
        train_Yvar = torch.from_numpy(Y_var_standardized).view((len(Y),1)).to(dtype=torch.float32)
        model = FixedNoiseGP(train_X = train_X, 
                             train_Y = train_Y, 
                             train_Yvar = train_Yvar,
                             covar_module = covar_module)
        
        # Set Prior on Homoscedastic Noise Model
        noise_prior = gpytorch.priors.GammaPrior(concentration = 1.1, rate = 0.05)
        noise_constraint = constraints.Positive()
        
        # Make Model Utilize Measured Y_var (Variance) = Heteroscedastic Noise Model
        model.likelihood = FixedNoiseGaussianLikelihood(noise = torch.from_numpy(Y_var_standardized).view((len(Y),1)).to(dtype=torch.float32).flatten(), 
                                                            learn_additional_noise=True,
                                                            noise_prior=noise_prior,
                                                            noise_constraint=noise_constraint)
        
        # Optimize Hyperparameters
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_model(mll)
        except:
            print('Problem w/ Hyper-Parameter Training: Attempting Again....')
            print()
            fit_gpytorch_model(mll) 
        
        #Base Acqusition Function
        AcqusitionFunction = qKnowledgeGradient(
        model = model,
        num_fantasies = num_fantasies,
        current_value = train_Y[train_X[...,-1]==0].max(),
        )
        
        # Initial Conditions
        batch_initial_conditions_tensor = gen_one_shot_kg_initial_conditions(
            acq_function=AcqusitionFunction,
            bounds=bounds,
            q = q_0 + q_1 + q_2,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            fixed_features=fixed_features_list_IC,
        )
        
        # Optimize Problem w/ fixed_features_list
        candidates, acq_value = optimize_acqf(
            acq_function = AcqusitionFunction,
            bounds = bounds,
            q = q_0 + q_1 + q_2,
            num_restarts = num_restarts,
            raw_samples = raw_samples,
            fixed_features = fixed_features_list,
            maximize = True,
            batch_initial_conditions = batch_initial_conditions_tensor,
        )
        
        # Define New Data
        X_0 = deepcopy(candidates[:q_0,:])
        X_1 = deepcopy(candidates[q_0:q_0 + q_1,:])
        X_2 = deepcopy(candidates[q_0 + q_1:q_0 + q_1 + q_2,:])
        
        # Define 0th Fidelity Simulation
        XX_0 = np.concatenate((X_0,
                               # X_0,
                               # X_0,
                               ))
        XX_0[:q_0,-1] = 0
        
        # Define 1st Fidelity Simulation
        XX_1 = np.concatenate((X_1,X_2))
        XX_1[:,-1] = 1
        
        # Define 2st Fidelity Simulation
        XX_2 = np.concatenate((X_1,X_2))
        XX_2[:,-1] = 2
        
        # Combine All Data
        X_new = np.concatenate((XX_0,XX_1,XX_2))
        Y_new = toyproblem(X_new)
        
        # Append to X and Y
        X = np.concatenate((X,X_new))
        Y = np.concatenate((Y,Y_new))
        
        #Get New Variance
        Y_var_standardized = ttb.get_variance_standardized(X,Y,toyproblem,n_reps)
        
        #Restandardize w/ New Experiments
        Y_standardized = (Y - Y.mean()) / Y.std()
        
        #Record Best Experiment
        Y_record[i,b+1] = Y[X[:,-1]==0].max()

print(Y_record)
np.save(file_name_output,Y_record)