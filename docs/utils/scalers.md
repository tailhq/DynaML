!!! summary
    Some attribute transformations are included in the DynaML distribution, here we show how to use them. All of them inherit `#!scala ReversibleScaler[I]` trait. They are contained in the `dynml.utils` package.


## Gaussian Centering

### Univariate

```scala
val mean = -1.5
val sigma = 2.5

val ugs = UnivariateGaussianScaler(mean, sigma)

val x = 3.0

val xs = ugs(x)

val xhat = ugs.i(xs)
```


### Multivariate

#### Diagonal

```scala
val mean: DenseVector[Double] = DenseVector(-1.5, 1.5, 0.25)
val sigma: DenseVector[Double] = DenseVector(0.5, 2.5, 1.0)

val gs = GaussianScaler(mean, sigma)

val x: DenseVector[Double] = DenseVector(0.2, -3.5, -1.5)

val xs = gs(x)

val xhat = gs.i(xs)
```

#### Full Matrix

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

```scala
val c = -1.5

val ums = UnivariateMeanScaler(c)

val x = 3.0

val xs = ums(x)

val xhat = ums.i(xs)
```

### Multivariate

```scala
val mean: DenseVector[Double] = DenseVector(-1.5, 1.5, 0.25)

val mms = MeanScaler(mean)

val x: DenseVector[Double] = DenseVector(0.2, -3.5, -1.5)

val xs = mms(x)

val xhat = mms.i(xs)
```


## Min-Max Centering

```scala
val min: DenseVector[Double] = DenseVector(-1.5, 1.5, 0.25)
val max: DenseVector[Double] = DenseVector(0.5, 2.5, 1.0)

val min_max_scaler = MinMaxScaler(min, max)

val x: DenseVector[Double] = DenseVector(0.2, -3.5, -1.5)

val xs = min_max_scaler(x)

val xhat = min_max_scaler.i(xs)
```


## Principal Component Analysis

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
