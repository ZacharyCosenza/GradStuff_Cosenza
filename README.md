# Instructions (NO LONGER MAINTAINED)

These are two python files that summarize the use of Bayesian Optimization
(BO) work during my PhD. I primarily use these methods to solve difficult problems
in cell culture media optimization. I would heavily recommend looking into the 
[botorch](https://botorch.org/) python package, and for more information
about customized kernels and Gaussian Process (GP) Models look into the [gpytorch](https://gpytorch.ai/)
package.

It is recommended to use a `conda` environment (via Anaconda) to download 
and manage the following python packages for `DBO_Solver.py` and `MOBO_Solver.py`. Use the most recent versions of each as the code is very general and shouldn't change much from version to version.

```
1. botorch
2. gpytorch
3. pytorch
4. scipy
```

Both files have test inputs *X* and outputs *Y* in as `numpy` arrays that can be replaced
with whatever data you need (imports from Excel or txt files for example). You should be 
able to run these files (and see some cool plots) using these test datapoints, or modify
to your liking.

# `DBO_Solver.py` (Desirability BO)

Using BO GP models I use monte-carlo to calculate a *desirability function* that can be 
modified to the users needs using various parameters. This function uses a stochastic
GP combined with a deterministic cost function (or any function) to quantify the quality
of a given set of datapoints *X*.

Nonconvex Optimization of Desirability Functions.” Quality Engineering 30(2): 293–310

# `MOBO_Solver.py` (Multi-Objective BO)

Again using BO GP models I solve for a multi-objective BO acquisition function: the
noisey expected hypervolume improvement function. Here quality is modeled as the amount
of output *Y* space dominated by a given *X*. Similar to `DBO_Solver.py` we use a stochastic
and deterministic function to model the output space, with the addition of a stochastic
constraint.