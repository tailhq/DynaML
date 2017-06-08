!!! summary
    Some attribute transformations are included in the DynaML distribution, here we show how to use them. All of them inherit `#!scala ReversibleScaler[I]` trait. They are contained in the `dynml.utils` package.


## Gaussian Centering

Gaussian scaling/centering involves calculating the sample mean and variance of data and applying a gaussian standardization operations using the calculated statistics.

It has different implementations in slightly varying contexts.

### Univariate

Univariate gaussian scaling involves

$$
\begin{align}
x &\in \mathbb{R} \\
\mu &\in \mathbb{R} \\
\sigma &\in \mathbb{R} \\
\bar{x} &= \frac{x-\mu}{\sigma}
\end{align}
$$

```scala
val mean = -1.5
val sigma = 2.5

val ugs = UnivariateGaussianScaler(mean, sigma)

val x = 3.0

val xs = ugs(x)

val xhat = ugs.i(xs)
```


### Multivariate

The data attributes form components of a vector, in this case we can assume each component is independent and calculate the diagonal variance or compute all the component covariances in the form of a symmetric matrix.

$$
\begin{align}
x &\in \mathbb{R}^n \\
\mu &\in \mathbb{R}^n \\
\Sigma &\in \mathbb{R}^{n \times n}\\
L L^\intercal &= \Sigma \\
\bar{x} &= L^{-1} (x - \mu)
\end{align}
$$


#### Diagonal

In this case the sample covariance matrix calculated from the data is diagonal and neglecting the correlations between the attributes.

$$
\Sigma = \begin{pmatrix}
\sigma^{2}_1 & \cdots & 0\\
 \vdots & \ddots  & \vdots\\
 0 & \cdots & \sigma^{2}_n  
\end{pmatrix}
$$


```scala
val mean: DenseVector[Double] = DenseVector(-1.5, 1.5, 0.25)
val sigma: DenseVector[Double] = DenseVector(0.5, 2.5, 1.0)

val gs = GaussianScaler(mean, sigma)

val x: DenseVector[Double] = DenseVector(0.2, -3.5, -1.5)

val xs = gs(x)

val xhat = gs.i(xs)
```

#### Full Matrix

When the sample covariance matrix is calculated taking into account correlations between data attributes.

```scala
val mean: DenseVector[Double] = DenseVector(-1.5, 1.5, 0.25)
val sigma: DenseMatrix[Double] = DenseMatrix(
  (2.5, 0.5, 0.25),
  (0.5, 3.5, 1.2),
  (0.25, 1.2, 2.25)
)

val mv_gs = MVGaussianScaler(mean, sigma)

val x: DenseVector[Double] = DenseVector(0.2, -3.5, -1.5)

val xs = mv_gs(x)

val xhat = mv_gs.i(xs)
```

## Mean Centering

### Univariate

$$
\begin{align}
x &\in \mathbb{R} \\
\mu &\in \mathbb{R} \\
\bar{x} &= x-\mu
\end{align}
$$


```scala
val c = -1.5

val ums = UnivariateMeanScaler(c)

val x = 3.0

val xs = ums(x)

val xhat = ums.i(xs)
```

### Multivariate

$$
\begin{align}
x &\in \mathbb{R}^n \\
\mu &\in \mathbb{R}^n \\
\bar{x} &= x - \mu
\end{align}
$$


```scala
val mean: DenseVector[Double] = DenseVector(-1.5, 1.5, 0.25)

val mms = MeanScaler(mean)

val x: DenseVector[Double] = DenseVector(0.2, -3.5, -1.5)

val xs = mms(x)

val xhat = mms.i(xs)
```


## Min-Max Scaling

Min-max scaling is also known as $0,1$ scaling because attributes are scaled down to the domain $[0, 1]$. This is done by calculating the minimum and maximum of attribute values.


```scala
val min: DenseVector[Double] = DenseVector(-1.5, 1.5, 0.25)
val max: DenseVector[Double] = DenseVector(0.5, 2.5, 1.0)

val min_max_scaler = MinMaxScaler(min, max)

val x: DenseVector[Double] = DenseVector(0.2, -3.5, -1.5)

val xs = min_max_scaler(x)

val xhat = min_max_scaler.i(xs)
```


## Principal Component Analysis

[Principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) consists of projecting data onto the eigenvectors of its sample covariance matrix.


```scala
val mean: DenseVector[Double] = DenseVector(-1.5, 1.5, 0.25)
val sigma: DenseMatrix[Double] = DenseMatrix(
  (2.5, 0.5, 0.25),
  (0.5, 3.5, 1.2),
  (0.25, 1.2, 2.25)
)

val pca = PCAScaler(mean, sigma)

val x: DenseVector[Double] = DenseVector(0.2, -3.5, -1.5)

val xs = pca(x)

val xhat = pca.i(xs)
```



!!! tip "Slicing scalers"

    It is possible to _slice_ the scalers shown above if they act on vectors. For example.
    ```scala
    //Slice on subset of columns
    val gs_sub: GaussianScaler = gs(0 to 1)
    //Slice on a single column
    val gs_last: UnivariateGaussianScaler = gs(2)
    ```
