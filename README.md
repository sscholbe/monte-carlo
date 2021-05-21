# Monte Carlo simulation for Credit Risk
High-performance Monte Carlo implementation to estimate credit risk. This is my attempt at the task presented by Deloitte in the course "Case Studies From Practice" in SS20.

### Challenge

The aim of this case study was to simulate one year default losses for a portfolio containing loans from different regions (CH, US and EU), with different credit ratings and exposures and to calculate loss risk measures. 

### Simulation

Over 100k scenarios (iterations):

1. Generate an observation for systematic risk factors corresponding to different regions
2. Generate an idiosyncratic risk factor realization for each loan
3. Determine the loan defaults and corresponding losses
4. Aggregate the losses

### Models

* The systematic risk factors are modeled as a multivariate normal distribution with a given correlation matrix
* The idiosyncratic risk factor realization is modeled as standard normal distribution

### Resulting distribution histogram over 100k iterations
![](/carlo/out/histogram.png?raw=true "Resulting distribution histogram")
