# Monte Carlo simulation for Credit Risk
High-performance Monte Carlo implementation to estimate the loss distribution of defaulting loans in a bank’s portfolio. This is my attempt at the task presented by Deloitte in the course "Case Studies From Practice" in SS20. Due to copyright reasons, I can't publish the `.csv` files with the data for the simulation.

## Challenge

The aim of this case study was to simulate one year default losses for a portfolio containing loans from different regions, with different credit ratings and exposures and to calculate loss risk measures. 

### Simulation

Over 100k scenarios (iterations):

1. Generate an observation for systematic risk factors corresponding to different regions
2. Generate an idiosyncratic risk factor realization for each loan (36k)
3. Determine the loan defaults and corresponding losses
4. Aggregate the losses

### Models

* The systematic risk factors are modeled as a multivariate normally distributed random variables with a given correlation matrix
* The idiosyncratic risk factor is modeled as standard normally distributed random variable

## Resulting distribution histogram
![](/carlo/out/histogram.png?raw=true "Resulting distribution histogram")

## Performance 

### How to make it run fast

The simulation is written in OpenCL for CPUs and GPUs. This language allows you to get the most out of your processor by paralellizing your computations nicely over many processing elements (PEs).

A Monte Carlo simulation is just repeating the same experiment over and over with almost no dependence between the iterations (except in this case a loss aggregation). This allows for heavy parallelization, which can be nicely done in OpenCL, where we can split up the iteration per PE. The scheduling is then done by the software and hardware.

### How to make it run very fast

A difficulty in implementing a Monte Carlo simulations in OpenCL is that OpenCL doesn't support random number generation naturally. But this is _the_ core of a Monte Carlo simulation (generating a random scenario over and over). One can either generate them on one core and feed them into the PEs (e.g. using streams) or go the extra mile and implement a PRNG that is of good quality and still fast, so I implemented [xoroshiro](https://prng.di.unimi.it/) that is used on each PE.

Since we need to generate standard normally distributed samples, but _xoroshiro_ only generates uniformly distributed samples, I'm using the [Box-Muller-transform](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform) to create two standard normally distributed samples out of two uniformly distributed samples.

For the generation of multivariate normally distributed samples, I'm computing the lower triangular matrix of the Cholesky decomposition of the covariance matrix once at startup. Multiplied with a standard normally distributed sample (generated using the previous methods), this produces a multivariate normally distributed sample with the given covariance. I compute the lower triangular matrix using the [Cholesky-Banachiewicz algorithm]( https://en.wikipedia.org/wiki/Cholesky_decomposition#The_Cholesky%E2%80%93Banachiewicz_and_Cholesky%E2%80%93Crout_algorithms).

### How to make it run absurdly fast

One kernel is responsible for one iteration of the simulation. There is enough work for one PE because the simulation requires to do computations for each of the 36k loans. The results (loss per iteration) is transferred back into the application's memory after the simulation.

Since we need one MVN sample per scenario (100k) and one UVN sample per loan (36k) per scenario (= 3.6bn UVN samples), the hottest part of the simulation is the generating the UVN idiosyncratic risk factor realization to then determine the default and losses. This is done using Box-Muller, but the loop (over the 36k loans) is unrolled twice because Box-Muller outputs two samples. Two accumulators for the aggregated losses are used to reduce the loop dependency.

I enable the OpenCL Fast Relaxed Math optimization flag, because all generated numbers should be sane. Also I compile the kernel at runtime, to make sure the  compiler can get the most out of its hardware.

