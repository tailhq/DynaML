
!!! summary
    The DynaML [`#! dynaml.probability.distributions`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/#io.github.mandar2812.dynaml.probability.distributions.package) package leverages and extends the `#! breeze.stats.distributions` package. Below is a list of distributions implemented.

## Specifying Distributions

Every probability density function $\rho(x)$ defined over some domain $x \in \mathcal{X}$ can be represented as $\rho(x) = \frac{1}{Z} f(x)$, where $f(x)$ is the un-normalized probability weight and $Z$ is the normalization constant. The normalization constant ensures that the density function sums to $1$ over the whole domain $\mathcal{X}$.

### Describing Skewness

An important analytical way to create skewed distributions was described by [Azzalani et. al](http://azzalini.stat.unipd.it/SN/skew-prop-aism.pdf). It consists of four components.

 * A symmetric probability density $\varphi(.)$
 * An odd function $w(.)$
 * A cumulative distribution function $G(.)$ of some symmetric density
 * A cut-off parameter $\tau$

$$
\rho(x) = \frac{1}{G(\tau)} \times \varphi(x)\times G(w(x) + \tau)
$$

## Distributions API

The `#!scala Density[T]` and `#!scala Rand[T]` traits form the API entry points for implementing probability distributions in breeze. In the `[#!scala dynaml.probability.distributions`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/#io.github.mandar2812.dynaml.probability.distributions.package) package, these two traits are inherited by [`#!scala GenericDistribution[T]`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/io/github/mandar2812/dynaml/probability/distributions/GenericDistribution) which is extended by [`#!scala AbstractContinuousDistr[T]`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/#io.github.mandar2812.dynaml.probability.distributions.AbstractContinuousDistr) and [`#!scala AbstractDiscreteDistr[T]`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/io/github/mandar2812/dynaml/probability/distributions/AbstractDiscreteDistr) classes.

!!! tip "Distributions which can produce confidence intervals"
    The trait [`#!scala HasErrorBars[T]`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/#io.github.mandar2812.dynaml.probability.distributions.HasErrorBars) can be used as a mix in to provide the ability of producing error bars to distributions. To extend it, one has to implement the `#!scala confidenceInterval(s: Double): (T, T)` method.

!!! tip "Skewness"
    The [`#!scala SkewSymmDistribution[T]`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/io/github/mandar2812/dynaml/probability/distributions/SkewSymmDistribution) class is the generic base implementations for skew symmetric family of distributions in DynaML.


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

$Z = {\frac {\Gamma_{p}\left({\frac {\nu +n+p-1}{2}}\right)}{(\pi )^{{\frac {np}{2}}}\Gamma _{p}\left({\frac {\nu +p-1}{2}}\right)}}|{\boldsymbol \Omega }|^{{-{\frac {n}{2}}}}|{\boldsymbol \Sigma }|^{{-{\frac {p}{2}}}}$

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


#### Univariate

$\mathcal{X} \equiv  \mathbb{R}$

$f(x) = \phi(\frac{x - \mu}{\sigma}) \Phi(\alpha (\frac{x-\mu}{\sigma}))$  

$Z = \frac{1}{2}$

$\phi()$ and $\Phi()$ being the standard gaussian density function and cumulative distribution function respectively


#### Multivariate

$\mathcal{X} \equiv  \mathbb{R}^d$

$f(x) = \phi_{d}(\mathbf{x}; \mathbf{\mu}, {\Sigma}) \Phi(\mathbf{\alpha}^{\intercal} L^{-1}(\mathbf{x} - \mathbf{\mu}))$  

$Z = \frac{1}{2}$

$\phi_{d}(.; \mathbf{\mu}, {\Sigma})$ and $\Phi()$ are the multivariate gaussian density function and standard gaussian univariate cumulative distribution function respectively and $L$ is the lower triangular Cholesky decomposition of $\Sigma$.

!!! note "Skewness parameter $\alpha$"
    The parameter $\alpha$ determines the skewness of the distribution and its sign tells us in which direction the distribution has a fatter tail. In the univariate case the parameter $\alpha$ is a scalar, while in the multivariate case $\alpha \in \mathbb{R}^d$, so for the multivariate skew gaussian distribution, there is a skewness value for each dimension.

*Usage*:
```scala
//Univariate
val mean = 1.5
val sigma = 1.5
val a = -0.5
val d = SkewGaussian(a, mean, sigma)

//Multivariate
val mu = DenseVector.ones[Double](4)
val alpha = DenseVector.fill[Double](4)(1.2)
val cov = DenseMatrix.eye[Double](4)*1.5
val md = MultivariateSkewNormal(alpha, mu, cov)
```

### Extended Skew Gaussian  

#### Univariate

The generalization of the univariate skew _Gaussian_ distribution.

$\mathcal{X} \equiv  \mathbb{R}$

$f(x) = \phi(\frac{x - \mu}{\sigma}) \Phi(\alpha (\frac{x-\mu}{\sigma}) + \tau\sqrt{1 + \alpha^{2}})$  

$Z = \Phi(\tau)$

$\phi()$ and $\Phi()$ being the standard gaussian density function and cumulative distribution function respectively

#### Multivariate

$\mathcal{X} \equiv  \mathbb{R}^d$

$f(x) = \phi_{d}(\mathbf{x}; \mathbf{\mu}, {\Sigma}) \Phi(\mathbf{\alpha}^{\intercal} L^{-1}(\mathbf{x} - \mathbf{\mu}) + \tau\sqrt{1 + \mathbf{\alpha}^{\intercal}\mathbf{\alpha}})$  

$Z = \Phi(\tau)$

$\phi_{d}(.; \mathbf{\mu}, {\Sigma})$ and $\Phi()$ are the multivariate gaussian density function and standard gaussian univariate cumulative distribution function respectively and $L$ is the lower triangular Cholesky decomposition of $\Sigma$.


*Usage*:
```scala
//Univariate
val mean = 1.5
val sigma = 1.5
val a = -0.5
val c = 0.5
val d = ExtendedSkewGaussian(c, a, mean, sigma)

//Multivariate
val mu = DenseVector.ones[Double](4)
val alpha = DenseVector.fill[Double](4)(1.2)
val cov = DenseMatrix.eye[Double](4)*1.5
val tau = 0.2
val md = ExtendedMultivariateSkewNormal(tau, alpha, mu, cov)
```

!!! warning "Confusing Nomenclature"
    The following distribution has a very similar form and name to the _extended skew gaussian_ distribution shown above. But despite its deceptively similar formula, it is a very different object.

    We use the name MESN to denote the variant below instead of its expanded form.

### MESN  

The  _Multivariate Extended Skew Normal_ or MESN distribution was formulated by [Adcock and Schutes](https://www.sheffield.ac.uk/polopoly_fs/1.137010!/file/Adcock-Skew-normal-exponential-.pdf). It is given by

$\mathcal{X} \equiv  \mathbb{R}^d$

$f(x) = \phi_{d}(\mathbf{x}; \mathbf{\mu} + \mathbf{\alpha}\tau, {\Sigma} + \mathbf{\alpha}\mathbf{\alpha}^\intercal) \Phi\left(\frac{\mathbf{\alpha}^{\intercal} \Sigma^{-1}(\mathbf{x} - \mathbf{\mu}) + \tau}{\sqrt{1 + \mathbf{\alpha}^{\intercal}\Sigma^{-1}\mathbf{\alpha}}}\right)$  

$Z = \Phi(\tau)$

$\phi_{d}(.; \mathbf{\mu}, {\Sigma})$ and $\Phi()$ are the multivariate gaussian density function and standard gaussian univariate cumulative distribution function respectively.

*Usage*:
```scala
//Univariate
val mean = 1.5
val sigma = 1.5
val a = -0.5
val c = 0.5
val d = UESN(c, a, mean, sigma)

//Multivariate
val mu = DenseVector.ones[Double](4)
val alpha = DenseVector.fill[Double](4)(1.2)
val cov = DenseMatrix.eye[Double](4)*1.5
val tau = 0.2
val md = MESN(tau, alpha, mu, cov)
```

!!! seealso "_Extended Skew Gaussian Process_ ESGP"
    The MESN distribution is used to define the finite dimensional probabilities for the [ESGP](/core/core_esgp.md) process.
