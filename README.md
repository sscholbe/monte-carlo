# Monte Carlo simulation for Credit Risk
This is a high-performance Monte Carlo implementation to estimate the loss distribution of defaulting loans in a bank’s portfolio and is my attempt at the task presented by Deloitte in the course "Case Studies From Practice" in SS20.

Due to copyright reasons, I can't publish the .csv files containing the input data for the simulation.
## Challenge
This case study aimed to simulate one-year default losses for a portfolio containing loans from different regions with different credit ratings and exposures and to calculate loss risk measures.
## Simulation
Over 100k scenarios (iterations):
1. Generate an observation for systematic risk factors corresponding to different regions.
2. Generate an idiosyncratic risk factor realization for each loan (36k).
3. Determine the loan defaults and corresponding losses.
4. Aggregate the losses.
## Models
The systematic risk factors are modeled as multivariate Gaussian random variables with a given correlation matrix.
The idiosyncratic risk factor is modeled as a standard univariate Gaussian random variable.
## Resulting distribution histogram
![](/carlo/out/histogram.png)
## Performance
### How to make it run fast
I wrote this simulation in OpenCL for CPUs and GPUs. This framework allows you to get the most out of your processor by letting you parallelize your computations  over many cores and processing elements (PEs).

A Monte Carlo simulation is repeating the same experiment repeatedly with almost no dependence between the iterations (except, in this case, a loss aggregation). This allows for heavy parallelization, which can be nicely done in OpenCL, where we can split up the iterations for each PE. The software and hardware then do the scheduling.
### How to make it run very fast
A difficulty in implementing Monte Carlo simulations in OpenCL is that OpenCL doesn't support random number generation naturally. But this is the core of a Monte Carlo simulation (generating a random scenario over and over). One can either generate them on one CPU core and feed them into the PEs (e.g., using streams) or go the extra mile and implement a PRNG on the PE, so I implemented [xoroshiro64](https://prng.di.unimi.it/).

Since we need to generate standard Gaussian samples, but xoroshiro only generates uniformly distributed samples, I'm using the [Box-Muller-transform](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform) to create two standard Gaussian samples out of two uniformly distributed samples.

For the generation of multivariate Gaussian samples, I'm computing the lower triangular matrix of the Cholesky decomposition of the covariance matrix once at startup in the application. Multiplying with a standard Gaussian sample (generated using the previous methods) on the PE produces a multivariate Gaussian sample with the given covariance. I compute the lower triangular matrix using the [Cholesky-Banachiewicz algorithm](https://en.wikipedia.org/wiki/Cholesky_decomposition#The_Cholesky%E2%80%93Banachiewicz_and_Cholesky%E2%80%93Crout_algorithms).
### How to make it run absurdly fast
One kernel is responsible for one iteration of the simulation. There is enough work for one PE because the simulation requires doing computations for each of the 36k loans. The results (loss per iteration) are transferred back into the application's memory after the simulation.

Since we need one MVN sample per scenario (100k) and one UVN sample per loan (36k) per scenario (= 3.6bn UVN samples), the hottest part of the simulation is generating the UVN idiosyncratic risk factor realization to then determine the default and losses. This is done using Box-Muller, but the loop (over the 36k loans) is unrolled twice because Box-Muller outputs two samples. Two accumulators for the aggregated losses are used to reduce the loop dependency.

I enable the OpenCL Fast Relaxed Math optimization flag because all generated numbers should be sane. Also, I compile the kernel at runtime to make sure the compiler can get the most out of its hardware.
### Benchmark
This is a table of the roughly the runtime every implementation has (should give you an overview); the last three just show how nicely an OpenCL implementation becomes faster with faster hardware as the framework is independent of the actual hardware.


|Programming language|Runtime|
|--------------------|-------|
|Naive Python|20 min|
|Naive Java|2 min|
|Parallel Java|30 s|
|C++|23 s|
|OpenCL (Intel HD Graphics 620)|800 ms|
|OpenCL (GTX 1060)|160 ms|
|OpenCL (RTX 2080)|60 ms|



