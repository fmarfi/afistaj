This process refers to a ==time series that displays a tendency to revert to its __historical mean value__. ==
Mathematically, such a (_continious_) __time series__ is referred to as an __Ornstein-Uhlenbeck process__.
- This is in contrast to a random walk.

# Testing for Mean Reversion
A _continous_ mean-reverting time series can be represented by an Ornstein-Ohlenbeck stochastic differential equation:
$$dx_{t}=\theta(\mu-x_{t})dt+\sigma dW_{t}
$$
Where $\theta$ is the rate of reversion to the mean, $\mu$ is the mean value of the process, $\sigma$ is the variance of the process and __$W_{t}$ is a Wiener process or Brownian Motion__.

- __Description__: Change of the price series in the next time period is proportional to the difference between the mean price and the current price with the _addition of Gaussian noise._
>[!Note]
>**dW_t is the instantaneous increment** of that process — the infinitesimal random "jump" over an infinitesimally small time step dt. It has mean 0 and variance dt, so its standard deviation scales as √dt. This √dt scaling is the single most important fact about stochastic calculus, and it's why Brownian paths look jagged rather than smooth: over a small step Δt, the deterministic drift term shrinks like Δt, but the random term only shrinks like √Δt — so as Δt → 0, randomness dominates the local behavior. This is also exactly why ordinary calculus (where you'd ignore second-order terms in Δt) doesn't work here, and why Itô calculus exists.

# Augmented Dickey Fuller (ADF) Test

The ADF is based on the idea of testing for the presence of a __unit root__ in an __autoregressive__ time series sample.

- It makes use of the fact that if a price series possesses mean reversion, then the next price level will be proportional to the current price level. 

A linear lag model of order p is used for the time series:
$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \delta_1 \Delta y_{t-1} + \cdots + \delta_{p-1} \Delta y_{t-p+1} + \epsilon_t$$
Where $\alpha$ is constant, $\beta$ represents the coefficient of a temporal trend and $\Delta y_t = y(t)-y(t-1)$. The role of the ADF hypothesis test is to consider the null hypothesis that $\gamma=0$, which would indicate (with $\alpha=\beta=0$) that the process is a random walk and thus non mean reverting.

## How is the ADF test carried out?
Calculate the __test statistic($DF_\tao$)__, which is given by the sample proportionality constant $\gamma$ divided by the standard error of the sample proportionality constant:
$$\begin{eqnarray}
DF_{\tau} = \frac{\hat{\gamma}}{SE(\hat{\gamma})}
\end{eqnarray}$$
- Dickey and Fuller have previously calculated the distribution of this test statistic, which allows us to determine the rejection of the hypothesis for any chosen percentage critical value.
- The ==test statistic is a negative number== and thus in order to be significant beyond the critical values, the number must be **more negative** than these values, i.e. __less than the critical values.__

A key practical issue for traders is that __any constant long-term drift in a price is of a _much smaller magnitude___ than __any short-term fluctuations__ and so the drift is often assumed to be zero ($\beta=0$) for the model.

>[!Note] 
>Since we are considering a lag model of order _p_, we need to actually set p to a particular value. It is usually sufficient, for trading research, to set _p=1_ to allow us to reject the null hypothesis.


- An alternative means of identifying a mean reverting time series is provided by the concept of __stationary__.


# Testing for Stationary
A time series(or stochastic process) is defined to be __strongly stationary__ if its _joint probability distribution_ is __invariant under translations in time or space__.
- The __mean and variance__ of the process do not change over time or space and they each do not follow a trend.

- A critical feature of __stationary price series__ is that the prices within the series diffuse from their initial value at a rate slower than that of a Geometric Brownian Motion.
- __By measuring the rate of this diffusive behaviour__ we can identify the nature of the time series.

# Hurst Exponent
The goal of the Hurst Exponent is to provide us with a scalar value that will help us to identify whether a series is __mean reverting__, __random walking__ or __trending__.

The _idea_ behind the Hurst Exponent calculation is that __we can use the variance of a log price series to assess the rate of diffusive behaviour__. For an arbitrary time lag $\tau$, the variance is given by:
$${\rm Var}(\tau) = \langle |\log(t+\tau)-\log(t)|^2 \rangle$$

Since we are comparing the rate of diffusion to that of a Geometric Brownian Motion, ==we can use the fact that at large $\tau$ we have that the variance is proportional to $\tau$ in the case of a GBM:==
$$\langle|\log(t+\tau)-\log(t)|^2 \rangle \sim \tau$$


The key insight is that __if any autocorrelations exist__ (i.e. any sequential price movements possess non-zero correlation) __then the above relationship is NOT valid__. Instead, it can be modified to _include an exponent value "2H"_, which gives us the Hurst Exponent value H:
$$
\langle |\log(t+\tau)-\log(t)|^2 \rangle \sim \tau^{2H}
$$

A time series can then be characterised in the following manner:
- $H<0.5$ - The time series is __mean reverting__.
- $H=0.5$ - The time series is a __Geometric Brownian Motion__.
- $H>0.5$ - The time series is __trending__.

In addition to characterisationn of the time series the Hurst Exponent also describes the extent to which a series behaves in the manner categorised. For instance, a value of H near 0 is highly mean reverting series, while for H near 1 the series is strongly trending.



[^II]: https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing/

