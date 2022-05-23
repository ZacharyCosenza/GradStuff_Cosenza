# Instructions for Download and Use (WORK IN PROGRESS)

These are two python files that summarize my ongoing use of Bayesian Optimization
(BO) work during my PhD. I primarily use these methods to solve difficult problems
in cell culture media optimization. I would heavily recommend looking into the 
[botorch](https://botorch.org/) python package, and for more information
about customized kernels and Gaussian Process (GP) Models look into the [gpytorch](https://gpytorch.ai/)
package.

It is recommended to use a `conda` environment (via Anaconda) to download 
and manage the following python packages for `DBO_Solver.py` and `MOBO_Solver.py`.

```
1. python 3.3.8 (>= 3.7 recommended)
2. botorch 0.3.3 (0.6.1.dev48+g812f08a3 for MOBO_Solver.py)
3. gpytorch 1.3.0
4. pytorch 1.7.1 (>= 1.9 recommended)
5. scipy 1.6.1
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