# Zachary Cosenza
# Botorch/Pytorch/Gpytorch General Toolbox

import numpy as np
from copy import deepcopy
import pyDOE as doe
from itertools import combinations
import torch
import gpytorch
from torch import Tensor
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from gpytorch import constraints
from botorch.sampling.samplers import MCSampler
from botorch.optim.fit import fit_gpytorch_scipy
from gpytorch.mlls import MarginalLogLikelihood
from typing import Any, Optional
from botorch.models.model import Model
from botorch.utils.transforms import (
    concatenate_pending_points,
    t_batch_mode_transform,
)
from botorch.acquisition.objective import MCAcquisitionObjective
from typing import Callable
import math
from botorch.acquisition import monte_carlo  # noqa F401
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective

def fit_gpytorch_multistart(
    mll: MarginalLogLikelihood, optimizer: Callable = fit_gpytorch_scipy, **kwargs: Any
) -> MarginalLogLikelihood:
    r"""
    This is a modified version of BoTorch `fit_gpytorch_model`. `fit_gpytorch_model`
    has some inconsistent behavior in fitting the embedding weights in LCEGP.
    The idea here is to get around this issue by aiming for a global fit.
    Args:
        mll: The marginal log-likelihood of the model. To be maximized.
        optimizer: The optimizer for optimizing the mll starting from an
            initialization of model parameters.
        **kwargs: Optional arguments.
    Returns:
        The optimized mll.
    """
    num_retries = kwargs.pop("num_retries",20)
    mll.train()
    original_state_dict = deepcopy(mll.model.state_dict())
    retry = 0
    state_dict_list = list()
    mll_values = torch.zeros(num_retries)
    max_error_tries = kwargs.pop("max_error_tries", 10)
    error_count = 0
    while retry < num_retries:
        #Get Parameters
        l = mll.model.covar_module.l
        s = mll.model.covar_module.s
        noise = mll.model.likelihood.second_noise_covar.noise
        new_l = torch.rand(l.shape) * 2
        new_s = torch.rand(s.shape) * 2
        new_noise = torch.rand(noise.shape,requires_grad=True) * 2
        mll.model.covar_module.l = torch.nn.Parameter(new_l,requires_grad = True)
        mll.model.covar_module.s = torch.nn.Parameter(new_s,requires_grad = True)
        mll.model.likelihood.second_noise_covar.noise = new_noise
        mll, info_dict = optimizer(mll, track_iterations=False, **kwargs)
        opt_val = info_dict["fopt"]
        if math.isnan(opt_val):
            if error_count < max_error_tries:
                error_count += 1
                continue
            else:
                state_dict_list.append(original_state_dict)
                mll_values[retry] = float("-inf")
                retry += 1
                continue

        # record the fitted model and the corresponding mll value
        state_dict_list.append(deepcopy(mll.model.state_dict()))
        mll_values[retry] = opt_val  # negate to get mll value
        retry += 1

    # pick the best among all trained models
    best_idx = mll_values.argmax()
    best_params = state_dict_list[best_idx]
    mll.model.load_state_dict(best_params)
    return mll.eval(),state_dict_list

class MultiIS_Kernel(gpytorch.kernels.Kernel):
    def __init__(self,num_IS):
        super(MultiIS_Kernel, self).__init__()
        self.num_IS = num_IS
        self.register_parameter(name="l", parameter=torch.nn.Parameter(torch.ones(num_IS)))
        self.register_parameter(name="s", parameter=torch.nn.Parameter(torch.ones(num_IS)))
        l_prior = gpytorch.priors.MultivariateNormalPrior(torch.ones(num_IS),torch.eye(num_IS)/2**2)
        s_prior = gpytorch.priors.MultivariateNormalPrior(torch.ones(num_IS),torch.eye(num_IS)/2**2)
        self.register_prior("l_prior",l_prior,"l")
        self.register_prior("s_prior",s_prior,"s")
        self.register_constraint(param_name = "l",
                                  constraint = constraints.Interval(0.001, 100))
        self.register_constraint(param_name = "s",
                                  constraint = constraints.Interval(0.001, 100))
    def forward(self,x1,x2,diag=False,last_dim_is_batch=False):
        num_IS = self.num_IS
        IS = x1[...,-1].long().unsqueeze(-1)
        L0 = self.l[0]
        x1_0 = x1[...,:-1].contiguous().div(L0)
        x2_0 = x2[...,:-1].contiguous().div(L0)
        d0 = torch.cdist(x1_0,x2_0).pow(2)
        K = self.s[0].pow(2) * torch.exp(-d0 / 2)
        for j in np.arange(num_IS-1):
            mask = (IS == j+1) * (IS == x2[...,-1].unsqueeze(-2))
            L = self.l[j+1]
            x1_j = x1[...,:-1].contiguous().div(L)
            x2_j = x2[...,:-1].contiguous().div(L)
            d_j = torch.cdist(x1_j,x2_j).pow(2)
            s = self.s[j+1].expand_as(K)
            K[mask] = K[mask] + s[mask].pow(2) * torch.exp(-d_j[mask] / 2)
        return K

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
        if type(DeterministicFunct) == MediaComplexityFunction:
            f_1 = - DeterministicFunct.forward(X,samples)
        else:
            f_1 = - DeterministicFunct(X,samples)
        
        # Stochastic Function 2 (Maximize This, botorch assumes maximization)
        f_2 = samples
        
        # Combine Functions for MultiObjective Case
        f_moo = torch.cat((f_2, f_1), axis = -1)
        
        return f_moo

class MediaComplexityFunction():
    
    def __init__(self,a,mn,stdev):
        super().__init__()
        self.a = a
        self.mn = mn
        self.stdev = stdev
        
    def forward(self,x,samples):
        a = self.a
        num_dim = x.shape[-1] - 1 #account for IS
        if samples.dim() == 3:
            X_ = x.unsqueeze(0).repeat(samples.shape[0],1,1)
        else:
            X_ = x.unsqueeze(0).repeat(samples.shape[0],1,1,1)
        phi_i = torch.exp(-0.5*(X_[...,:-1]/a)**2)
        phi = num_dim - torch.sum(phi_i,dim = -1)
        f = phi.unsqueeze(-1)
        return (f - self.mn) / self.stdev

def Compute_Complexity_Media(x,a):
    num_dim = x.shape[-1] - 1 #account for IS
    phi_i = np.exp(-0.5*(x[...,:-1]/a)**2)
    phi = num_dim - np.sum(phi_i,axis = -1)
    return phi

def MediaCostFunction(x,samples):
    num_dim = x.shape[-1] - 1 #account for IS
    c = torch.tensor(np.array([1.0E-08, #MEM-NEAA
                        1.0E-08,#MEM-EAA
                        1.0E-08,#MEM-Vitamin
                        1.0E-08,#Salts
                        1.0E-08,#Trace Metals
                        1.0E-08,#DNA Precursor
                        1.0E-08,#Fatty Acids
                        1.0E-08,#Sodium Selenite
                        1.0E-08,#Ascorbic Acid
                        1.0E-08,#Glucose
                        1.0E-08,#Glutamine
                        1.0E-08,#Sodium Pyruvate
                        1.0E-08,#Sodium Chloride
                        0.030447656,#Insulin
                        0.003889299,#Transferrin
                        0.633400136,#FGF2
                        0.088898265,#TGFb1
                        0.003583711,#EGF
                        1.0E-08,#Progesterone
                        1.0E-08,#Estradiol
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
    c = np.array([1.0E-08,
                        1.0E-08,
                        1.0E-08,
                        1.0E-08,
                        1.0E-08,
                        1.0E-08,
                        1.0E-08,
                        1.0E-08,
                        1.0E-08,
                        1.0E-08,
                        1.0E-08,
                        1.0E-08,
                        1.0E-08,
                        0.030447656,
                        0.003889299,
                        0.633400136,
                        0.088898265,
                        0.003583711,
                        1.0E-08,
                        1.0E-08,
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
    c = 1.0 #unstandardized constant to get greater than
    con = samples - (c - mu0) / std0
    return - con.squeeze(-1) # negative = feasible

def get_training_data(N,N_0,px,toyproblem,num_IS):
    #N number of low fidelity points per fidelity
    #N_0 number of high fidelity points (must be less than N)
    #px dimensionality of problem (not including fidelity parameter)
    #toyproblem name of problem
    #num_IS number of information sources
    X = doe.lhs(px, samples = N)
    XX = X[np.random.choice(N,N_0, replace=False),:]
    X = np.repeat(X,num_IS - 1,axis=0)
    A = np.tile((np.arange(num_IS - 1)+1).reshape(-1,1),(N,1))
    X = np.hstack((X,A))
    XX = np.hstack((XX,np.zeros((N_0,1)).reshape(N_0,1)))
    X = np.vstack((X,XX))
    Y = toyproblem(X)
    return X,Y
    
def get_variance_standardized(X,Y,toyproblem,n_reps):
    Y_var = np.zeros(X.shape[0])
    Y = np.zeros(X.shape[0])
    Y_list = []
    for i in np.arange(X.shape[0]):
        S = toyproblem(np.tile(X[i,:],(n_reps,1)))
        Y_list.append(S)
    common_mean = np.mean(Y_list)
    common_std = np.std(Y_list)
    Y_list_ = (Y_list - common_mean) / common_std
    for i in np.arange(X.shape[0]):
        Y_var[i] = Y_list_[i].var()
        Y[i] = Y_list_[i].mean()
    return Y_var

def get_further_exp(x_opt,q,q0,acq_function):
    combo_list = list(combinations(np.arange(q),q0))
    results = np.zeros(len(combo_list))
    for i in np.arange(len(combo_list)):
        results[i] = acq_function(x_opt[combo_list[i],:])
    return results,combo_list

def print_parameters(model):
    for param_name, param in model.named_parameters():
        print(param_name)
        print(param)

class RegularPosterior(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model, sampler=sampler, objective=objective)
    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)
        #need to squeeze out last dimension for obj to make sense
        obj = self.objective(samples, X=X).squeeze(-1)
        post_D = obj.squeeze(-1)
        return post_D.mean(dim=0)

def get_borehole(x):
    #Initalize
    y = np.zeros(x.shape[0])
    rw=x[:,0]*(0.15-0.05)+0.05
    r=x[:,1]*(50000-100)+100
    Tu=x[:,2]*(115600-63070)+63070
    Hu=x[:,3]*(1110-990)+990
    Tl=x[:,4]*(116-63.1)+63.1
    Hl=x[:,5]*(820-700)+700
    L=x[:,6]*(1680-1120)+1120
    Kw=x[:,7]*(12045-9855)+9855
    #Get Fidelities
    ind_1 = x[:,-1] == 1
    ind_2 = x[:,-1] == 2
    #Underlying Function
    y = 2*np.pi*Tu*(Hu-Hl)/np.log(r/rw)/(1+2*L*Tu/(np.log(r/rw)*rw**2*Kw)+Tu/Tl)
    #Add Fidelity 1
    common_mean = 100
    common_error = 0.1
    y[ind_1] = y[ind_1] + np.random.uniform(-common_mean*common_error,common_mean*common_error,size = len(y[ind_1]))
    #Add Fidleity 2
    y[ind_2] = 5*Tu[ind_2]*(Hu[ind_2]-Hl[ind_2])/np.log(r[ind_2]/rw[ind_2])/(1.5+2*L[ind_2]*Tu[ind_2]/(np.log(r[ind_2]/rw[ind_2])*rw[ind_2]**2*Kw[ind_2])+Tu[ind_2]/Tl[ind_2])
    return y

#Sphere Function Bias
def get_sphere(x):
    #Get Fidelities
    ind_0 = x[:,-1] == 0
    ind_1 = x[:,-1] == 1
    ind_2 = x[:,-1] == 2
    y = np.zeros(x.shape[0])
    y[ind_0] = np.sum((x[ind_0,:-1]-0.5)**2,axis = 1) 
    y[ind_1] = np.sum((x[ind_1,:-1]-0.4)**2,axis = 1) 
    y[ind_2] = np.sum((x[ind_2,:-1]-0.75)**2,axis = 1) 
    y = y + np.random.uniform(-0.2,0.2,size=len(y))
    y[ind_1] = 3 * y[ind_1] + (x[ind_1,1] - x[ind_1,3] + x[ind_1,9]) + np.random.uniform(-0.1,0.1,size = len(y[ind_1]))
    y[ind_2] = y[ind_2] + (x[ind_2,0] - x[ind_2,4] + x[ind_2,8]) - x[ind_2,0] * x[ind_2,4]
    return -y

#Trid Function Bias
def get_trid(x):
    #Get Fidelities
    ind_1 = x[:,-1] == 1
    ind_2 = x[:,-1] == 2
    d = x.shape[1]-1
    A = np.sum((x[:,:-1]-1)**2,axis = 1)
    B =  np.zeros(x.shape)
    for i in np.arange(start=1,stop=d):
        B[:,i] = x[:,i]*x[:,i-1]
    y = A + np.sum(B,axis = 1)
    y = y + np.random.uniform(-0.1,0.1,size = len(y))
    y[ind_1] = y[ind_1] - 2 *(x[ind_1,2] - x[ind_1,5] + x[ind_1,9]) + 1 + np.random.uniform(-0.1,0.1,size = len(y[ind_1]))
    y[ind_2] = y[ind_2] + 2 *(x[ind_2,0] - x[ind_2,4] + x[ind_2,8]) + x[ind_2,0] * x[ind_2,4]
    return y

# Bowl-Line Function Bias
def get_bowlline(x):
    #Get Fidelities
    ind_0 = x[:,-1] == 0
    ind_1 = x[:,-1] == 1
    ind_2 = x[:,-1] == 2
    c = np.array([0,0.5,0.3,0.8,1])
    y = np.zeros(x.shape[0])
    y[ind_0] = np.sum((x[ind_0,:5] - c)**2,axis = 1) + np.sum(x[ind_0,5:-1],axis = 1)
    y[ind_1] = np.sum((x[ind_1,:5] - c - 0.35)**2,axis = 1) + np.sum(x[ind_1,5:-1],axis = 1)
    y[ind_2] = np.sum((x[ind_2,:5] - c + 0.35)**2,axis = 1) + np.sum(x[ind_2,5:-1],axis = 1)
    y = y + np.random.uniform(-0.35,0.35,size = len(y))
    y[ind_1] = y[ind_1] - 2 *(x[ind_1,2] - x[ind_1,5] + x[ind_1,9]) + 1 + np.random.uniform(-0.35,0.35,size = len(y[ind_1]))
    y[ind_2] = y[ind_2] + 2 * (x[ind_2,0] + x[ind_2,4] - x[ind_2,8]) + 2 * x[ind_2,0] * x[ind_2,4] - 2 * x[ind_2,1] * x[ind_2,2]
    return y

# Bowl-Line-Hard Function Bias
def get_bowlline_hard(x):
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