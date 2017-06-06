
### Extract Data as Univariate Time Series

```scala
extractTimeSeries(Tfunc)
```

* _Type_: `#!scala DataPipe[Stream[String], Stream[(Double, Double)]]`
* _Result_: This pipe assumes its input to be of the form `YYYY,Day,Hour,Value`. It takes as input a function (TFunc) which converts a `#!scala (Double, Double, Double)` into a single _timestamp_ like value. The pipe processes its data source line by line and outputs a `#!scala Tuple2` in the following format `(Timestamp,Value)`.

### Extract data as Multivariate Time Series

```scala
extractTimeSeriesVec(Tfunc)
```

* _Type_: `#!scala DataPipe[Stream[String], Stream[(Double, DenseVector[Double])]]`
* _Result_: This pipe is similar to `#!scala extractTimeSeries` but for application in multivariate time series analysis such as nonlinear autoregressive models with exogenous inputs. The pipe processes its data source line by line and outputs a `#!scala (Double, DenseVector[Double])` in the following format `(Timestamp,Values)`.


### Construct Time differenced Data

```scala
deltaOperation(deltaT, timelag)
```

* _Type_: `#!scala DataPipe[Stream[(Double, Double)], Stream[(DenseVector[Double], Double)]]`
* _Result_: In order to generate features for auto-regressive models, one needs to construct sliding windows in time. This function takes two parameters `#!scala deltaT`: the auto-regressive order and `#!scala timelag`: the time lag after which the windowing is conducted. E.g Let `#!scala deltaT = 2` and `#!scala timelag = 1` This pipe will take stream data of the form $(t, y(t))$ and output a stream which looks like $(t, [y(t-2), y(t-3)])$


### Construct multivariate Time differenced Data

```scala
deltaOperationVec(deltaT: Int)
```

* _Type_: `#!scala DataPipe[Stream[(Double, Double)], Stream[(DenseVector[Double], Double)]]`
* _Result_: A variant of `#!scala deltaOperation` for NARX models.


### Haar Discrete Wavelet Transform

```scala
haarWaveletFilter(order: Int)
```

* _Type_: `#!scala DataPipe[DenseVector[Double], DenseVector[Double]]`
* _Result_: A Haar Discrete wavelet transform.
