
!!! summary
    The DynaML `#! dynaml.probability.distributions` package leverages and extends the `#! breeze.stats.distributions` package. Below is a list of distributions implemented.

## Specifying Distributions

Every probability density function $\rho(x)$ defined over some domain $x \in \mathcal{X}$ can be represented as $\rho(x) = \frac{1}{Z} f(x)$, where $f(x)$ is the un-normalized probability weight and $Z$ is the normalization constant. The normalization constant ensures that the density function sums to $1$ over the whole domain $\mathcal{X}$.

### Describing Skewness

An important analytical way to create skewed distributions was described by [Azzalani et. al](http://azzalini.stat.unipd.it/SN/skew-prop-aism.pdf). It consists of four components.

 * A symmetric probability density $\varphi(.)$
 * An odd function $w()$
 * A cumulative distribution function $G(.)$ of some symmetric density
 * A cut-off parameter $\tau$

$$
\rho(x) = \frac{1}{G(\tau)} \times \varphi(x)\times G(w(x) + \tau)
$$

## Distributions API

The `#!scala Density[T]` and `#!scala Rand[T]` traits form the API entry points for implementing probability distributions in breeze. In the `#!scala dynaml.probability.distributions` package, these two traits are inherited by `#!scala GenericDistribution[T]` which is extended by `#!scala AbstractContinuousDistr[T]` and `#!scala AbstractDiscreteDistr[T]` classes.

!!! tip "Distributions which can produce confidence intervals"
    The trait `#!scala HasErrorBars[T]` can be used as a mix in to provide the ability of producing error bars to distributions. To extend it, one has to implement the `#!scala confidenceInterval(s: Double): (T, T)` method.

!!! tip "Skewness"
    The `#!scala SkewSymmDistribution[T]` class is the generic base implementations for skew symmetric family of distributions in DynaML.


## Distributions Library

Apart from the distributions defined in the `#!scala breeze.stats.distributions`, users have access to the following distributions implemented in the `#!scala dynaml.probability.distributions`.

### Multivariate Students T  

Defines a _Students' T_ distribution over the domain of finite dimensional vectors.

$\mathcal{X} \equiv  \mathbb{R}^{n}$

$f(x) = \left[1+{\frac {1}{\nu }}({\mathbf {x} }-{\boldsymbol {\mu }})^{\rm {T}}{\boldsymbol {\Sigma }}^{-1}({\mathbf {x} }-{\boldsymbol {\mu }})\right]^{-(\nu +p)/2}$  

$Z = \frac{\Gamma \left[(\nu +p)/2\right]}{\Gamma (\nu /2)\nu ^{p/2}\pi ^{p/2}\left|{\boldsymbol {\Sigma }}\right|^{1/2}}$

*Usage*:
```scala
val mu = 2.5
val mean = DenseVector(1.0, 0.0)
val cov = DenseMatrix((1.5, 0.5), (0.5, 2.5))
val d = MultivariateStudentsT(mu, mean, cov)
```

### Matrix T  

Defines a _Students' T_ distribution over the domain of matrices.

$\mathcal{X} \equiv  \mathbb{R}^{n \times p}$

$f(x) = \left|{\mathbf {I}}_{n}+{\boldsymbol \Sigma }^{{-1}}({\mathbf {X}}-{\mathbf {M}}){\boldsymbol \Omega }^{{-1}}({\mathbf {X}}-{\mathbf {M}})^{{{\rm {T}}}}\right|^{{-{\frac {\nu +n+p-1}{2}}}}$  

$Z = {\frac {\Gamma _{p}\left({\frac {\nu +n+p-1}{2}}\right)}{(\pi )^{{\frac {np}{2}}}\Gamma _{p}\left({\frac {\nu +p-1}{2}}\right)}}|{\boldsymbol \Omega }|^{{-{\frac {n}{2}}}}|{\boldsymbol \Sigma }|^{{-{\frac {p}{2}}}}$

*Usage*:
```scala
val mu = 2.5
val mean = DenseMatrix((-1.5, -0.5), (3.5, -2.5))
val cov_rows = DenseMatrix((1.5, 0.5), (0.5, 2.5))
val cov_cols = DenseMatrix((0.5, 0.1), (0.1, 1.5))
val d = MatrixT(mu, mean, cov_rows, cov_cols)
```

### Matrix Normal  

Defines a _Gaussian_ distribution over the domain of matrices.

$\mathcal{X} \equiv  \mathbb{R}^{n \times p}$

$f(x) = \exp\left( -\frac{1}{2} \, \mathrm{tr}\left[ \mathbf{V}^{-1} (\mathbf{X} - \mathbf{M})^{T} \mathbf{U}^{-1} (\mathbf{X} - \mathbf{M}) \right] \right)$  

$Z = (2\pi)^{np/2} |\mathbf{V}|^{n/2} |\mathbf{U}|^{p/2}$

*Usage*:
```scala
val mean = DenseMatrix((-1.5, -0.5), (3.5, -2.5))
val cov_rows = DenseMatrix((1.5, 0.5), (0.5, 2.5))
val cov_cols = DenseMatrix((0.5, 0.1), (0.1, 1.5))
val d = MatrixNormal(mean, cov_rows, cov_cols)
```

### Truncated Normal  

Defines a univariate _Gaussian_ distribution that is defined in a finite domain.

$\mathcal{X} \equiv  [a, b]$

$f(x) = \begin{cases} \phi ({\frac {x-\mu }{\sigma }}) & a \leq x \leq b\\0 & else\end{cases}$  

$Z = \sigma \left(\Phi ({\frac {b-\mu }{\sigma }})-\Phi ({\frac {a-\mu }{\sigma }})\right)$

$\phi()$ and $\Phi()$ being the gaussian density function and cumulative distribution function respectively

*Usage*:
```scala
val mean = 1.5
val sigma = 1.5
val (a,b) = (-0.5, 2.5)
val d = TruncatedGaussian(mean, sigma, a, b)
```

### Skew Gaussian  

The univariate skew _Gaussian_ distribution.

$\mathcal{X} \equiv  \mathbb{R}$

$f(x) = \phi(\frac{x - \mu}{\sigma}) \Phi(\alpha (\frac{x-\mu}{\sigma}))$  

$\phi()$ and $\Phi()$ being the standard gaussian density function and cumulative distribution function respectively

$Z = \frac{1}{2}$

*Usage*:
```scala
val mean = 1.5
val sigma = 1.5
val a = -0.5
val d = SkewGaussian(a, mean, sigma)
```

### Extended Skew Gaussian  

The generalization of the univariate skew _Gaussian_ distribution.

$\mathcal{X} \equiv  \mathbb{R}$

$f(x) = \frac{1}{\Phi(\tau)} \phi(\frac{x - \mu}{\sigma}) \Phi(\alpha (\frac{x-\mu}{\sigma}) + \tau\sqrt{1 + \alpha^{2}})$  

$\phi()$ and $\Phi()$ being the standard gaussian density function and cumulative distribution function respectively

*Usage*:
```scala
val mean = 1.5
val sigma = 1.5
val a = -0.5
val c = 0.5
val d = ExtendedSkewGaussian(c, a, mean, sigma)
```
