!!! summary

    The `#!scala utils` object contains some useful helper functions which are used by a number of API components of DynaML.



## String/File Processing

### Load File into a Stream

```scala
val content = utils.textFileToStream("data.csv")
```

### String Replace

Replace all occurrences of a string (or regular expression) in a target string

```scala
val new_str = utils.replace(find = ",")(replace = "|")(input = "1,2,3,4")

```

### URL download

Download the content of a url to a specified location on disk.

```scala
utils.downloadURL("www.google.com", "google_home_page.html")
```

### Write to File

```scala

val content: Stream[String] = _
utils.writeToFile("foo.csv")(content)
```

## Numerics

### `#!scala log1p`

Calculates $log_{e}(1+x)$.

```scala
val l = utils.log1pExp(0.02)
```

### Haar DWT Matrix

Constructs the Haar [discrete wavelet transform](https://en.wikipedia.org/wiki/Discrete_wavelet_transform) matrix for orders which are powers of two.

```scala
val dwt_mat = utils.haarMatrix(math.pow(2, 3).toInt)
```

### Hermite Polynomials

The [Hermite polynomials](https://en.wikipedia.org/wiki/Hermite_polynomials) are an important class of orthogonal polynomials used in numerical analysis. There are two definitions of the _Hermite_ polynomials i.e. the probabilist and physicist definitions, which are equivalent up-to a scale factor. The the `utils` object, the probabilist polynomials are calculated.

```scala

//Calculate the 3rd order Hermite polynomial

val h3 = (x: Double) => utils.hermite(3, x)

h3(2.5)

```

### Chebyshev Polynomials

[Chebyshev polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials) are another important class of orthogonal polynomials used in numerical analysis. There are two types, the _first kind_ and _second kind_.


```scala

//Calculate the Chebyshev polynomial of second kind order 3

val c23 = (x: Double) => utils.chebyshev(3, x, kind = 2)

c23(2.5)

```


### Quick Select

The quick select aims to find the $k^{th}$ smallest element of a list of numbers.

```scala
val second = utils.quickselect(List(3,2,4,5,1,6), 2)
```

### Median

```scala
val second = utils.median(List(3,2,4,5,1,6))
```

### Sample Statistics

Calculate the mean and variance (or covariance), minimum, maximum of a list of `#!scala DenseVector[Double]` instances.

```scala
val data: List[DenseVector[Double]] = _

val (mu, vard): (DenseVector[Double], DenseVector[Double]) =
  utils.getStats(data)

val (mean, cov): (DenseVector[Double], DenseMatrix[Double]) =
  utils.getStatsMult(data)

val (min, max) = utils.getMinMax(data)
```
