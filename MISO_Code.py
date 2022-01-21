# Zachary Cosenza
# Code for MISO Publication

# Import Packages
import csv
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import torch
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim.optimize import optimize_acqf
from itertools import combinations
import gpytorch
from torch import Tensor
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from gpytorch import constraints
from botorch.sampling.samplers import MCSampler
from botorch.fit import _set_transformed_inputs
from botorch.optim.fit import fit_gpytorch_scipy
from gpytorch.mlls import MarginalLogLikelihood
from typing import Any, Optional
from botorch.models.model import Model
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)
from botorch.models.gp_regression import FixedNoiseGP
from botorch.acquisition.objective import MCAcquisitionObjective
from typing import Callable
from botorch.acquisition.objective import IdentityMCObjective
import inspect
import warnings
import math
from botorch import settings
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import SamplingWarning
from botorch.sampling.samplers import IIDNormalSampler
from torch.quasirandom import SobolEngine
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood

def get_cost_of_media(x):
    r"""
    This gives you some cost c for x input, used in media optimization hence
    the name.
    Args:
        x: batch_shape x n x px+1, which is input
    Returns:
        cost : batch_shape x n, which is cost of input X
    """
    c = torch.from_numpy(np.array([0.006528867,
                                    0.0143325,
                                    6.43125E-09,    
                                    9.83828E-06,
                                    0.22325625,
                                    0.020872579,
                                    4.9392,
                                    14,
                                    1.09433E-05,
                                    0.007219333,
                                    3.98125E-07,
                                    1.62823E-06,
                                    6.13219E-06,
                                    0.000604333])).to(dtype = torch.float32)
    cost_min = 0.020607015
    cost = cost_min + torch.matmul(x[...,:-1],c)
    return cost

def get_media_cost_objective(samples,X,common_mean,common_std):
    r"""
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
    cost = get_cost_of_media(X)
    cost_min = 0.020607015
    cost_max = 0.020607015 + 19.23264982
    cost_s = 1 #degree of importance (set at 1 as default)
    
    #Normalize Cost
    d_cost = ((cost - cost_max) / (cost_min - cost_max)).pow(cost_s)
    if d_cost.dim() == 1:
        d_cost = d_cost.view(samples.shape[1],len(d_cost))
        
    #Restandardize f(x)
    samples = samples * common_std + common_mean
    
    #f(x) (Maximize)
    f_min = 0.5
    f_max = 2
    f_s = 1 #degree of importance (set at 1 as default)
    mask = samples < f_min
    d_f = ((samples - f_min) / (f_max - f_min)).pow(f_s)
    d_0 = torch.zeros(1).expand_as(d_f)
    d_f[mask] = d_0[mask]
    
    #Desirability Function
    equation = 'kbij,bi->kbij'
    D = (torch.einsum(equation,d_f,d_cost) + 0.1) **(0.5)
    return D

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
        #Get Parameters from Module (must be modified if you mess with kernel params)
        l = mll.model.covar_module.l
        s = mll.model.covar_module.s
        noise = mll.model.likelihood.second_noise_covar.noise
        new_l = torch.rand(l.shape) * 2 #initalize in region [0,2]
        new_s = torch.rand(s.shape) * 2 #initalize in region [0,2]
        new_noise = torch.rand(noise.shape,requires_grad=True) * 2
        mll.model.covar_module.l = torch.nn.Parameter(new_l,requires_grad = True)
        mll.model.covar_module.s = torch.nn.Parameter(new_s,requires_grad = True)
        mll.model.likelihood.second_noise_covar.noise = new_noise #b/c we have hetero noise, must init this as well
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
    _set_transformed_inputs(mll=mll)
    return mll.eval()

class MultiIS_ARD_Kernel(gpytorch.kernels.Kernel):
    r"""Multi-Information Source ARD Kernel

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
    def __init__(self,px,num_IS):
        super(MultiIS_ARD_Kernel, self).__init__()
        
        # Set Design Parameters
        self.num_IS = num_IS #number of information sources
        self.px = px #number of dimensions
        
        # Set Length-Scale Hyperparameter l
        self.register_parameter(name="l", parameter=torch.nn.Parameter(torch.ones(num_IS*px)))
        l_prior = gpytorch.priors.MultivariateNormalPrior(torch.ones(num_IS*px),torch.eye(num_IS*px)/2**2)
        self.register_prior("l_prior",l_prior,"l")
        self.register_constraint(param_name = "l",
                                  constraint = constraints.Interval(0.01, 10))
        
        # Set Output-Scale Hyperparameter s
        self.register_parameter(name="s", parameter=torch.nn.Parameter(torch.ones(num_IS)))
        s_prior = gpytorch.priors.MultivariateNormalPrior(torch.ones(num_IS),torch.eye(num_IS)/2**2)
        self.register_prior("s_prior",s_prior,"s")
        self.register_constraint(param_name = "s",
                                  constraint = constraints.Interval(0.01, 10))
        
    def forward(self,x1,x2,diag=False,last_dim_is_batch=False):
        
        # Find IS for Each Component in (batch_shape x n)
        IS = x1[...,-1].long().unsqueeze(-1)
        
        # Squared Exponential Kernel for All Points
        L0 = self.l[0:self.px]
        x1_0 = x1[...,:-1].contiguous().div(L0)
        x2_0 = x2[...,:-1].contiguous().div(L0)
        d0 = torch.cdist(x1_0,x2_0).pow(2)
        K = self.s[0].pow(2) * torch.exp(-d0 / 2)
        
        for j in np.arange(self.num_IS-1):
            
            # Squared Exponential Kernel for Mask
            # Mask : all non-0 points x1,x2 pairs that share an IS from 1...num_IS
            mask = (IS == j+1) * (IS == x2[...,-1].unsqueeze(-2))
            L = self.l[(j+1)*self.px:(j+1)*self.px+self.px]
            x1_j = x1[...,:-1].contiguous().div(L)
            x2_j = x2[...,:-1].contiguous().div(L)
            d_j = torch.cdist(x1_j,x2_j).pow(2)
            s = self.s[j+1].expand_as(K)
            
            # Kernel is Additive
            K[mask] = K[mask] + s[mask].pow(2) * torch.exp(-d_j[mask] / 2)
            
        return K

class CostMCObjective(MCAcquisitionObjective):
    r"""Objective generated from a generic callable.

    Allows to construct arbitrary MC-objective functions from a generic
    callable. In order to be able to use gradient-based acquisition function
    optimization it should be possible to backpropagate through the callable.

    Example:
        >>> generic_objective = GenericMCObjective(
                lambda Y, X: torch.sqrt(Y).sum(dim=-1),
            )
        >>> samples = sampler(posterior)
        >>> objective = generic_objective(samples)
    """

    def __init__(self, objective: Callable[[Tensor, Tensor], Tensor]) -> None:
        r"""Objective generated from a generic callable.

        Args:
            objective: A callable `f(samples, X)` mapping a
                `sample_shape x batch-shape x q x m`-dim Tensor `samples` and
                an optional `batch-shape x q x d`-dim Tensor `X` to a
                `sample_shape x batch-shape x q`-dim Tensor of objective values.
        """
        super().__init__()
        if len(inspect.signature(objective).parameters) == 1:
            warnings.warn(
                "The `objective` callable of `GenericMCObjective` is expected to "
                "take two arguments. Passing a callable that expects a single "
                "argument will result in an error in future versions.",
                DeprecationWarning,
            )

            def obj(samples: Tensor, X: Tensor) -> Tensor:
                return objective(samples)

            self.objective = obj
        else:
            self.objective = objective
            
    def setcustomparams(self,common_mean,common_std):
        #needed for restandardization
        self.common_mean = common_mean
        self.common_std = common_std

    def forward(self, samples: Tensor, X: Tensor) -> Tensor:
        r"""Evaluate the feasibility-weigthed objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x m`-dim Tensors of
                samples from a model posterior.
            X: A `batch_shape x q x d`-dim tensor of inputs. Relevant only if
                the objective depends on the inputs explicitly.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of objective values
            weighted by feasibility (assuming maximization).
        """
        #this is not available in pytorch so i placed custom params (which are needed
        # in the restandardization code, into forward here)
        common_mean = self.setcustomparams[0]
        common_std = self.setcustomparams[1]
        return self.objective(samples,X,common_mean,common_std)

class qNoisyCostyExpectedImprovement(MCAcquisitionFunction):
    r"""MC-based batch Noisy Expected Improvement.

    This function does not assume a `best_f` is known (which would require
    noiseless observations). Instead, it uses samples from the joint posterior
    over the `q` test points and previously observed points. The improvement
    over previously observed points is computed for each sample and averaged.

    `qNEI(X) = E(max(max Y - max Y_baseline, 0))`, where
    `(Y, Y_baseline) ~ f((X, X_baseline)), X = (x_1,...,x_q)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qNEI = qNoisyExpectedImprovement(model, train_X, sampler)
        >>> qnei = qNEI(test_X)
        
        Note from Zac: this is a modified qNEI code to allow for input X
    """

    def __init__(
        self,
        model: Model,
        X_baseline: Tensor,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        X_pending: Optional[Tensor] = None,
        prune_baseline: bool = False,
        **kwargs: Any,
    ) -> None:
        r"""q-Noisy Expected Improvement.

        Args:
            model: A fitted model.
            X_baseline: A `batch_shape x r x d`-dim Tensor of `r` design points
                that have already been observed. These points are considered as
                the potential best design point.
            sampler: The sampler used to draw base samples. Defaults to
                `SobolQMCNormalSampler(num_samples=512, collapse_batch_dims=True)`.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points
                that have points that have been submitted for function evaluation
                but have not yet been evaluated. Concatenated into `X` upon
                forward call. Copied and set to have no gradient.
            prune_baseline: If True, remove points in `X_baseline` that are
                highly unlikely to be the best point. This can significantly
                improve performance and is generally recommended. In order to
                customize pruning parameters, instead manually call
                `botorch.acquisition.utils.prune_inferior_points` on `X_baseline`
                before instantiating the acquisition function.
        """
        super().__init__(
            model=model, sampler=sampler, objective=objective, X_pending=X_pending
        )
        #this speeds up code but origionally didn't allow for inputs X (which were needed
        # becuase the initialization strategy by botorch needed to have random starts at 0th
        # fidelity)
        if prune_baseline:
            X_baseline = prune_inferior_points_with_X(
                model=model,
                X=X_baseline,
                objective=objective,
            )
        self.register_buffer("X_baseline", X_baseline)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qNoisyExpectedImprovement on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Noisy Expected Improvement values at the
            given design points `X`, where `batch_shape'` is the broadcasted batch shape
            of model and input `X`.
        """
        q = X.shape[-2]
        X_full = torch.cat([X, match_batch_shape(self.X_baseline, X)], dim=-2)
        # TODO: Implement more efficient way to compute posterior over both training and
        # test points in GPyTorch (https://github.com/cornellius-gp/gpytorch/issues/567)
        posterior = self.model.posterior(X_full)
        samples = self.sampler(posterior)
        #need to squeeze out last dimension for obj to make sense
        obj = self.objective(samples, X=X_full).squeeze(-1)
        diffs = obj[..., :q].max(dim=-1).values - obj[..., q:].max(dim=-1).values
        return diffs.clamp_min(0).mean(dim=0)

def prune_inferior_points_with_X(
    model: Model,
    X: Tensor,
    objective: Optional[MCAcquisitionObjective] = None,
    num_samples: int = 2048,
    max_frac: float = 1.0,
    sampler: Optional[MCSampler] = None,
) -> Tensor:
    r"""Prune points from an input tensor that are unlikely to be the best point.

    Given a model, an objective, and an input tensor `X`, this function returns
    the subset of points in `X` that have some probability of being the best
    point under the objective. This function uses sampling to estimate the
    probabilities, the higher the number of points `n` in `X` the higher the
    number of samples `num_samples` should be to obtain accurate estimates.

    Args:
        model: A fitted model. Batched models are currently not supported.
        X: An input tensor of shape `n x d`. Batched inputs are currently not
            supported.
        objective: The objective under which to evaluate the posterior.
        num_samples: The number of samples used to compute empirical
            probabilities of being the best point.
        max_frac: The maximum fraction of points to retain. Must satisfy
            `0 < max_frac <= 1`. Ensures that the number of elements in the
            returned tensor does not exceed `ceil(max_frac * n)`.
        sampler: If provided, will use this customized sampler instead of
            automatically constructing one with `num_samples`.

    Returns:
        A `n' x d` with subset of points in `X`, where

            n' = min(N_nz, ceil(max_frac * n))

        with `N_nz` the number of points in `X` that have non-zero (empirical,
        under `num_samples` samples) probability of being the best point.
    """
    if X.ndim > 2:
        # TODO: support batched inputs (req. dealing with ragged tensors)
        raise UnsupportedError(
            "Batched inputs `X` are currently unsupported by prune_inferior_points"
        )
    max_points = math.ceil(max_frac * X.size(-2))
    if max_points < 1 or max_points > X.size(-2):
        raise ValueError(f"max_frac must take values in (0, 1], is {max_frac}")
    with torch.no_grad():
        posterior = model.posterior(X=X)
    if sampler is None:
        if posterior.event_shape.numel() > SobolEngine.MAXDIM:
            if settings.debug.on():
                warnings.warn(
                    f"Sample dimension q*m={posterior.event_shape.numel()} exceeding "
                    f"Sobol max dimension ({SobolEngine.MAXDIM}). Using iid samples "
                    "instead.",
                    SamplingWarning,
                )
            sampler = IIDNormalSampler(num_samples=num_samples)
        else:
            sampler = SobolQMCNormalSampler(num_samples=num_samples)
    samples = sampler(posterior)
    if objective is None:
        objective = IdentityMCObjective()
    obj_vals = objective(samples,X).squeeze(1).squeeze(-1)
    if obj_vals.ndim > 2:
        # TODO: support batched inputs (req. dealing with ragged tensors)
        raise UnsupportedError(
            "Batched models are currently unsupported by prune_inferior_points"
        )
    is_best = torch.argmax(obj_vals, dim=-1)
    idcs, counts = torch.unique(is_best, return_counts=True)

    if len(idcs) > max_points:
        counts, order_idcs = torch.sort(counts, descending=True)
        idcs = order_idcs[:max_points]

    return X[idcs] 

def get_further_exp(x_opt,q,q0,acq_function):
    combo_list = list(combinations(np.arange(q),q0))
    results = np.zeros(len(combo_list))
    for i in np.arange(len(combo_list)):
        results[i] = acq_function(x_opt[combo_list[i],:])
    return results,combo_list

def get_data(file,xsize,ysize):
    doc = open(file,'r')
    f = csv.reader(doc,delimiter='\t')
    out = np.zeros([xsize,ysize])
    i = 0
    for line in f:
        out[i,:] = line
        i = i + 1
    return out

# Import Data
X = get_data('BO_data.txt',209,15) # X
Outputs = get_data('BO_outputs.txt',209,2) # X
Y = Outputs[:,0]
Y_var = Outputs[:,1]
    
# Model Parameters
common_mean = 0.929625502 #mean of raw data
common_std = 0.570520559 #std of raw data
num_IS = 4 #number of information sources
px = 14 #number of components
q_high = 1 #number of experiments to be designed at high fidelity
q_low = 3 #number of experiments to be designed at low fidelity
q = q_high + q_low

# List Names of X Components in Order
labels = ['T','I','SS','AA','Glu','Gluta','Albu',
          'FBS','H','D','P','Esd','Ethan','Glutath']

# Reformat Data
X = torch.from_numpy(X).to(dtype = torch.float32) #needs to be tensor
Y = torch.from_numpy(Y).to(dtype = torch.float32) #needs to be tensor
Y_var = torch.from_numpy(Y_var).to(dtype = torch.float32) #needs to be tensor
X_ = X.view((1,X.shape[0],X.shape[1])) #reshape data
Y_ = Y.view((1,len(Y),1)) #reshape data
Y_var_= Y_var.view((len(Y),1)) #reshape data

# Set Covariance Type
covar_module = MultiIS_ARD_Kernel(px,num_IS)

# Set Model Type
model = FixedNoiseGP(train_X = X_, 
                      train_Y = Y_, 
                      train_Yvar = Y_var_,
                      covar_module = covar_module)

# Set Prior on Homoscedastic Noise Model
noise_prior = gpytorch.priors.GammaPrior(concentration = 1.1, rate = 0.05)
noise_constraint = constraints.Positive()

# Make Model Utilize Measured Y_var (Variance) = Heteroscedastic Noise Model
model.likelihood = FixedNoiseGaussianLikelihood(noise = Y_var.flatten(), 
                                                    learn_additional_noise=True,
                                                    noise_prior=noise_prior,
                                                    noise_constraint=noise_constraint)

# Optimize Hyperparameters
mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_multistart(mll) #multistart

# Print Hyperparameters
with torch.no_grad():
    for param_name, param in model.named_parameters():
        print(param_name)
        print(param)

#Set Up Monte Carlo Sampler
MC_SAMPLES = 2000
bounds = torch.tensor([[0.0] * (px+1), [1.0] * (px+1)]) #bounds should be [0,1]
bounds[-1,-1] = 1.0 * (num_IS - 1) #last bound should allow for fidelity indicator, not needed usually
qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

#Set Up Cost Function Objective
cost_aware_objective = CostMCObjective(get_media_cost_objective)
cost_aware_objective.setcustomparams = common_mean,common_std
X_0 = X[X[...,-1]==0,:] #baseline should be only highest fidelity experiments

# Set Up Optimization Objective
aq_cost = qNoisyCostyExpectedImprovement(model = model, 
                                        X_baseline = X_0, 
                                        sampler = qmc_sampler,
                                        objective = cost_aware_objective)

# Optimize Problem
x_opt , y_opt = optimize_acqf(acq_function=aq_cost,
                                bounds=bounds,
                                q=q,
                                num_restarts=25,
                                raw_samples=1000,
                                fixed_features={px: 0})

#Apply Choice Rule
results,combo_list = get_further_exp(x_opt,q,q_low,aq_cost)
combo_opt = combo_list[np.argmax(results)]

#Attach Conditions to Optimal X and Y
X_highfid = deepcopy(x_opt[combo_opt,:])
X_lowfid = deepcopy(x_opt)

#Generate Testing Data for Visualization
n = 300
x = torch.rand((n,px+1))
x[:,-1] = 0
y_pred = model.posterior(x).mean.detach().numpy()

plt.figure(figsize = (8,4))
plt.subplot(1,2,1)
plt.plot(x[:,7],y_pred.flatten() * common_std + common_mean,'k.',alpha = 0.5,label = '$y_{predicted}$')
plt.plot(X[:,7,],Y * common_std + common_mean,'rs',alpha = 0.5,label = '$y_{data}$')
plt.xlabel('$x_7$')
plt.ylabel('$y$')
plt.legend(frameon = True)

x_un = torch.ones((1,px+1)) * 0.5
x_un[:,-1] = 0
cost_un = np.zeros(n)
y_un = np.zeros(n)
des_un = np.zeros(n)
qei = np.zeros(n)
qei_cost = np.zeros(n)

for i in np.arange(n):
    x_un[0,7] = i/n
    cost_un[i] = get_cost_of_media(x_un)
    posterior_predict = model.posterior(x_un).mean.detach()
    #change to have mc dimension
    posterior_predict = posterior_predict.unsqueeze(0)
    des_un[i] = get_media_cost_objective(posterior_predict, x_un, common_mean, common_std)

plt.subplot(1,2,2)
plt.plot(np.linspace(0,1,n),des_un,'k',alpha = 0.5)
plt.xlabel('$x_7$')
plt.ylabel('Objective Function $a(x)$')
plt.subplots_adjust(wspace=0.30)

plt.figure(figsize = (10,4))
for i in np.arange(px):
    plt.subplot(2,7,i+1)
    x_un = np.ones(shape = (n,px+1)) * 0.5
    x_un[:,-1] = 0
    x_un[:,i] = np.linspace(0,1,n)
    x_un = torch.tensor(x_un).to(dtype=torch.float32)
    x_un_ = x_un.clone().detach()
    x_un_ = x_un.reshape((1,n,px+1)).to(dtype=torch.float32)
    y_pred_un = model.posterior(x_un_).mean.detach().numpy()
    plt.plot(x_un[:,i],y_pred_un.flatten(),'k',label = 'Prediction')
    plt.vlines(X_lowfid[:,i],y_pred_un.min(),y_pred_un.max(),'r',label = 'High Fidelity Optima')
    plt.vlines(X_highfid[:,i],y_pred_un.min(),y_pred_un.max(),'b',label = 'Low Fidelity Optima')
    plt.xticks([])
    plt.yticks([])
    plt.title(labels[i])
plt.legend(frameon = False,bbox_to_anchor=(1.0, 0.5))