# Hypothesis testing with Bayesian AR(p) models
This project focuses on hypothesis testing. In particular, we test how many Auto Regressive (AR) coefficients are necessary to represent a time series. We will dive into the hypothesis testing as proposed in "Probability theory: the logic of science" by E.T. Jaynes. Jaynes gives us a logically consistent way to test hypotheses. In finding the order of a AR process, each order, <img alt="$p$" src="https://github.com/RobRomijnders/bayesian_model_comparison/blob/master/svgs/2ec6e630f199f589a2402fdf3e0289d5.svg" align="middle" width="8.239720500000002pt" height="14.102549999999994pt"/>, will represent a hypothesis. The hypothesis test from Jaynes book will tell us for which hypothesis we have most evidence.

# Hypothesis testing
For Jaynes, all inferences in probability theory stem from the product and sum rule. For those who want to go deeper, Jaynes even derivates these rules from three elementary axioms. Here, we are interested in the probability of a hypothesis to be true. We find this probability of the hypothesis using the observed data and the prior assumptions we make. Let the data be <img alt="$D$" src="https://github.com/RobRomijnders/bayesian_model_comparison/blob/master/svgs/78ec2b7008296ce0561cf83393cb746d.svg" align="middle" width="14.013780000000002pt" height="22.381919999999983pt"/> and our background information be <img alt="$I$" src="https://github.com/RobRomijnders/bayesian_model_comparison/blob/master/svgs/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg" align="middle" width="8.484300000000001pt" height="22.381919999999983pt"/>, then the product rule follows:

<img alt="$p(H|DI) = p(H|I)\frac{p(D|HI)}{p(D|I)}$" src="https://github.com/RobRomijnders/bayesian_model_comparison/blob/master/svgs/ddf0ddadfda459534a8c779ba8245cf9.svg" align="middle" width="186.20959499999998pt" height="33.14091000000001pt"/>

In the case, we compare two hypotheses against eachother, then the odds of one hypothesis over another is:

<img alt="$\frac{p(H_1|DI)}{p(H_2|DI)} = \frac{p(H_1|I)p(D|H_1I)p(D|I)}{p(H_2|I)p(D|H_2I)p(D|I)} = \frac{p(H_1|I)p(D|H_1I)}{p(H_2|I)p(D|H_2I)}$" src="https://github.com/RobRomijnders/bayesian_model_comparison/blob/master/svgs/aaa902dc18f81da5e23150c327628260.svg" align="middle" width="347.99622pt" height="33.14091000000001pt"/>

Jaynes likes to express probability ratios in decibels. So our equation follows to:

<img alt="$10\log_{10} \frac{p(H_1|DI)}{p(H_2|DI)} =  10\log_{10} \frac{p(H_1|I)}{p(H_2|I)} + 10\log_{10} \frac{p(D|H_1I)}{p(D|H_2I)}$" src="https://github.com/RobRomijnders/bayesian_model_comparison/blob/master/svgs/1703383162964466370a5928aad49569.svg" align="middle" width="379.59949499999993pt" height="33.14091000000001pt"/>

We give all hypotheses equal prior probability. Therefore, we will focus in the evidence terms: <img alt="$p(D|H_iI)$" src="https://github.com/RobRomijnders/bayesian_model_comparison/blob/master/svgs/d34c8931216b2dbee79eaa23f5f28e94.svg" align="middle" width="67.11704999999999pt" height="24.56552999999997pt"/>

# Our Auto Regressive model
We set out to find the order of auto regressive coefficients to represent our time series. Finding this order is usefull for filtering or prediction of stationary signals. One example could be to predict the next time step in stock or pricing series.

The model for timestep, <img alt="$t$" src="https://github.com/RobRomijnders/bayesian_model_comparison/blob/master/svgs/4f4f4e395762a3af4575de74c019ebb5.svg" align="middle" width="5.9139630000000025pt" height="20.14650000000001pt"/> in a time series is:

<img alt="$y_t = \sum_{i=0}^{p-1} \alpha_i y_{t-i} + e_t$" src="https://github.com/RobRomijnders/bayesian_model_comparison/blob/master/svgs/dfb3678c84e2f37c20598295a3328b78.svg" align="middle" width="157.511145pt" height="31.305779999999984pt"/>

where <img alt="$e \sim \mathcal{N}(0, \sigma^2)$" src="https://github.com/RobRomijnders/bayesian_model_comparison/blob/master/svgs/a212d3550e8486da7774275008c5d146.svg" align="middle" width="90.89223pt" height="26.70657pt"/>

## Probability of the data
From our model, we write down the likelihood of the parameters: (from here on, we will take the conditioning on background <img alt="$I$" src="https://github.com/RobRomijnders/bayesian_model_comparison/blob/master/svgs/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg" align="middle" width="8.484300000000001pt" height="22.381919999999983pt"/> implicit)

p(y_t|\alpha H) = \mathcal{N}(\sum_{i=0}^{p-1} \alpha_i y_{t-i}, \sigma^2)

Or, lumping it all together: (for simplicity, we allow for negative indices in <img alt="$y$" src="https://github.com/RobRomijnders/bayesian_model_comparison/blob/master/svgs/deceeaf6940a8c7a5a02373728002b0f.svg" align="middle" width="8.616960000000002pt" height="14.102549999999994pt"/>)

<img alt="$p(y| \alpha H) = (2 \pi \sigma^2)^{-\frac{T}{2}}  e^{\frac{-1}{2\sigma^2}\sum_{t=0}^{T}(\sum_{i=0}^{p-1}\alpha_iy_{t-i}-y_t)^2}$" src="https://github.com/RobRomijnders/bayesian_model_comparison/blob/master/svgs/245d19afd2ad4c8fb8dae708506f5430.svg" align="middle" width="333.81859499999996pt" height="36.93723pt"/>
<img alt="$\log p(y| \alpha H) = -\frac{T}{2}\log (2 \pi \sigma^2)  - \frac{1}{2\sigma^2}\sum_{t=0}^{T}(\sum_{i=0}^{p-1}\alpha_iy_{t-i}-y_t)^2$" src="https://github.com/RobRomijnders/bayesian_model_comparison/blob/master/svgs/f8a69c9081245802c1904c8ddf36508f.svg" align="middle" width="433.679895pt" height="32.19743999999999pt"/>

# Approximations to the marginal likelihood
Now we are stuck. In the final equation, we wrote down the probability of the data. However, we use a parameter to produce this probability. To answer our hypothesis test, we look for the probability of the data, conditioned only on the hypothesis: <img alt="$p(y|H)$" src="https://github.com/RobRomijnders/bayesian_model_comparison/blob/master/svgs/0604346078319f52228de750dd105499.svg" align="middle" width="49.094595pt" height="24.56552999999997pt"/>. We also name this probability <img alt="$p(y|H)$" src="https://github.com/RobRomijnders/bayesian_model_comparison/blob/master/svgs/0604346078319f52228de750dd105499.svg" align="middle" width="49.094595pt" height="24.56552999999997pt"/> sometimes the marginal likelihood.

As the name suggests, we can find out marginal likelihood by marginalising the likelihood:

<img alt="$p(y|H) = \int_\alpha  p(y\alpha|H)\delta \alpha= \int_\alpha  p(y|H\alpha) p(\alpha|H)\delta \alpha$" src="https://github.com/RobRomijnders/bayesian_model_comparison/blob/master/svgs/e1b140d73f48c66c3e6dc0b09732da82.svg" align="middle" width="339.72559499999994pt" height="26.48447999999999pt"/>

For this marginalisation, we discuss three approximations:

  * Monte Carlo approximation: approximate the integral with samples from <img alt="$p(\alpha)$" src="https://github.com/RobRomijnders/bayesian_model_comparison/blob/master/svgs/047e1a09e399f0d8794e25397993769c.svg" align="middle" width="31.514670000000002pt" height="24.56552999999997pt"/>
  * Laplace approximation: approximate the integral with a Laplace approximation around the Maximum A Posteriori (MAP) value for <img alt="$\alpha$" src="https://github.com/RobRomijnders/bayesian_model_comparison/blob/master/svgs/c745b9b57c145ec5577b82542b2df546.svg" align="middle" width="10.537065000000004pt" height="14.102549999999994pt"/>
  * Bayesian Information criterion: approximate the integral with the MAP value plus a penalty term for the free parameters that were fitted

## Monte Carlo approximation
Instead of evaluating the integral, we may rewrite is as:

<img alt="$p(y|H) =  \int_\alpha  p(y|H\alpha) p(\alpha|H)\delta \alpha = E_{alpha\sim p(\alpha|H)}[p(y|H\alpha)]$" src="https://github.com/RobRomijnders/bayesian_model_comparison/blob/master/svgs/f9ca5ecd31f1d4044020dac20d849c48.svg" align="middle" width="407.806245pt" height="26.48447999999999pt"/>

We approximate the expectation with the Monte Carlo estimate:

<img alt="$p(y|H) \simeq \frac{1}{M} \sum_{m=0}^{M-1} p(y|\alpha_m H) \ ; \ \alpha_m\sim p(\alpha|H)$" src="https://github.com/RobRomijnders/bayesian_model_comparison/blob/master/svgs/bad43cf2fd4d8eafbc598f8ed74724d9.svg" align="middle" width="329.078145pt" height="32.19743999999999pt"/>

## Laplace approximation
In the Laplace approximation, we fit a Gaussian distribution around the maximum, like so:

p(y|H) =  \int_\alpha  p(y|H\alpha) p(\alpha|H)\delta \alpha = p(y|\alpha_{MAP}H)p(\alpha_{MAP}|H)(\det A \frac{1}{2\pi})^{-\frac{1}{2}}

We approximate the intergrand with a Gaussian, because can integrate the Gaussian analytically.

## Bayesian Information Criterion
In the final approximation, we don't use a property of the integral, but penalize complexity of our model explicitly.

That is, we approximate the marginal likelihood like:
p(y|H) \simeq p(y|H \alpha_{MAP}) - \frac{\text{dof}}{2} \log (N)

Here the <img alt="$\text{dof}$" src="https://github.com/RobRomijnders/bayesian_model_comparison/blob/master/svgs/5fac9a364e06b4458a4da879590d7810.svg" align="middle" width="23.564805pt" height="22.745910000000016pt"/> indicates the _degrees of freedom_ in the model. In our case, the degrees of freedom equals the number of AR coefficients



# Results
In the code, we implement all three approximations. Below we show the table that rolls out:

![table](doc/im/table.png)

Two observations:

  * First, the numerical values do not equal for the various approximations. After all, we make approximations.
  * Second, the highest marginal likelihood occurs at AR(3) for all three approximations. Fortunately, because we used an AR(3) to generate our data!


# Further reading

  * [Probability theory by E.T. Jaynes ](https://www.cambridge.org/core/books/probability-theory/9CA08E224FF30123304E6D8935CF1A99)
  * [On auto regressive models](https://www.pearson.com/us/higher-education/program/Kay-Fundamentals-of-Statistical-Processing-Volume-I-Estimation-Theory/PGM50476.html)
  * [Chapter 28 on Model comparison of Mackay's Information theory, inference and learning algorithms.](http://www.inference.org.uk/itila/book.html)


