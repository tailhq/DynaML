---
title: Process Time Indexed Data
keywords: sample
summary: "This page outlines the various library workflows which apply to pre-processing of data for time series analyses."
sidebar: product1_sidebar
tags: [pipes, workflow]
permalink: p1_sample4.html
folder: product1
---

## Time Series Data

### Extract Data as Univariate Time Series

```scala
extractTimeSeries(Tfunc)
```

* _Type_: ```DataPipe[Stream[String], Stream[(Double, Double)]] ```
* _Result_: This pipe assumes its input to be of the form `YYYY,Day,Hour,Value`. It takes as input a function (TFunc) which converts a ```(Double, Double, Double)``` into a single "timestamp" like value. The pipe processes its data source line by line and outputs a ```Tuple2``` in the following format `(Timestamp,Value)`.

### Extract data as Multivariate Time Series

```scala
extractTimeSeriesVec(Tfunc)
```

* _Type_: ```DataPipe[Stream[String], Stream[(Double, DenseVector[Double])]] ```
* _Result_: This pipe is similar to ```extractTimeSeries``` but for application in multivariate time series analysis such as nonlinear autoregressive models with exogenous inputs. The pipe processes its data source line by line and outputs a ```(Double, DenseVector[Double])``` in the following format `(Timestamp,Values)`.


### Construct Time differenced Data

```scala
deltaOperation(deltaT, timelag)
```

* _Type_: ```DataPipe[Stream[(Double, Double)], Stream[(DenseVector[Double], Double)]] ```
* _Result_: In order to generate features for auto-regressive models, one needs to construct sliding windows in time. This function takes two parameters `deltaT`: the auto-regressive order and `timelag`: the time lag after which the windowing is conducted. E.g Let deltaT = 2 and timelag = 1 This pipe will take stream data of the form $$(t, y(t))$$ and output a stream which looks like $$(t, [y(t-2), y(t-3)])$$


### Construct multivariate Time differenced Data

```scala
deltaOperationVec(deltaT: Int)
```

* _Type_: ```DataPipe[Stream[(Double, Double)], Stream[(DenseVector[Double], Double)]] ```
* _Result_: A variant of `deltaOperation` for NARX models.


### Haar Discrete Wavelet Transform

```scala
haarWaveletFilter(order: Int)
```

* _Type_: ```DataPipe[DenseVector[Double], DenseVector[Double]] ```
* _Result_: A Haar Discrete wavelet transform.

{% include links.html %}
