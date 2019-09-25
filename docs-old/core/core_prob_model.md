<br/>

DynaML takes a sampling first approach to a probability API and so there are many operations and
transformations that can easily be applied to random variables created in the DynaML REPL. _Probability models_ enable
the user to express multivariate distributions in terms of conditional probability factorizations.

$$
p(x,\theta) = p(x|\theta) \times p(\theta) = p(\theta|x) \times p(x)
$$

Conditional probability factorizations are at the center of _Bayes Theorem_

$$
p(\theta|x) = \frac{p(x|\theta) \times p(\theta)}{p(x)}
$$

In Bayesian analysis, $$p(\theta)$$ is known as the _prior probability_ or just _prior_ while
$$
p(x|\theta)
$$ is called the _data likelihood_ or just _likelihood_. The _prior_ distribution encodes our belief about which areas of the parameter space are more likely. The _likelihood_ distribution states how likely it is for some data $$x$$ is produced for a specific value of parameters $$\theta$$.

## Probability Models

In DynaML the ```ProbabilityModel``` class can be used to create arbitrary kinds of conditional probability factorizations, for example consider the simple _Beta-Bernoulli_ coin toss model.

The _Beta-Bernoulli_ model can be specified as follows.

**Prior**

The _prior_ is a _Beta_ distribution with parameters $$\alpha, \beta$$.

$$
\begin{align}
  p(\theta) &= \frac{1}{B(\alpha, \beta)} \theta^{\alpha - 1} (1 - \theta)^{\beta - 1} \\
  B(\alpha, \beta) &= \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha + \beta)}
\end{align}
$$

**Likelihood**

For a value $$\theta$$ sampled from the _prior_ distribution, we generate $$n$$ coin tosses with probability of heads for each toss being $$\theta$$. The _Binomial_ distribution gives the probability that out of such $$n$$ loaded coin tosses; $$k$$ tosses will turn up heads.

$$
\begin{align}
p(x = k|\theta;n) = \binom{n}{k}\theta^k (1 - \theta)^{n-k}
\end{align}
$$

### Creation

```scala
//Start with a beta prior
val p = RandomVariable(new Beta(7.5, 7.5))

//Simulate 500 coin tosses with probability of heads; p ~ Beta(7.5, 7.5)
val coinLikelihood = DataPipe((p: Double) => new BinomialRV(500, p))

//Construct a probability model.
val c_model = ProbabilityModel(p, coinLikihood)

```

### Prior Sampling

We can now visualize the prior.

```scala
histogram((1 to 2000).map(_ => p.sample()))
```

![histogram](/images/histogram-prior.png)


### Posterior Sampling

The ```ProbabilityModel``` class has a built in value member called ```posterior```, which is an instance of the ```RandomVariable``` class. It can be used to sample from the posterior distribution of the model parameters given a set of data observations. In the _Beta-Bernoulli_ coin toss example, we created a likelihood model that was a _Binomial distribution_ over 500 coin tosses.

To generate samples from the posterior distribution, we must provide _data_ or in our case; the number of heads observed in 500 coin tosses.

```scala

// The posterior distribution for the situation
// when 350 heads are observed out of 500 coin tosses
val post = c_model.posterior(350)
val postSample = (1 to 2000).map(_ => post.sample())
//Generate samples from posterior and visualize as a histogram.
hold()
histogram(postSample)
unhold()
```

![histogram](/images/histogram-post-prior.png)


From the posterior samples generated we can now examine sufficient statistics, such as the posterior mean. From Bayesian theory it is known that for a _Beta-Bernoulli_ model, the posterior is another _Beta_ distribution specified by
$$
p(\theta|x = k) = Beta(\alpha + k, \beta + n - k)
$$. From the properties of the _Beta_ distribution the mean is given in our case by $$\frac{\alpha+k}{\alpha + \beta + n}$$ giving a value of about $$0.694174$$. We can verify this from out posterior sample.

```scala
val postMean = postSample.sum/postSample.length

postMean: Double = 0.6851085955685318
```
