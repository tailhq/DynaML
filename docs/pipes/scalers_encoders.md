---
title: Reversible Transformations
---

!!! sumary
    The DynaML pipes API is extended to represent reversible transformations for encode-decode and scaling tasks

Reversible transformations are quite important in computational sciences (Machine Learning included), some examples include _auto-encoders_, _min-max scaling_, _gaussian scaling_ etc. In DynaML they occupy a special place in the pipes API.

## Scalers and Reversible Scalers

It is quite common in data processing operations to carry out transformations on the data such as feature scaling, wavelet transform, Fourier transform etc. The ```Scaler[S]``` trait represents all such transformations (by extending ```DataPipe[Source, Target]```) which map values from a type ```S``` to itself.

The trait ```ReversibleScaler[S]``` extends ```Scaler[S]``` by adding the inverse transformation (```i```) as a value member. Some common implementations of the reversible scaler type are.

**Gaussian Feature Scaling**: Implemented in the ```GaussianScaler``` case class which takes two arguments $$\mu$$ and $$\sigma$$ as instances of breeze ```DenseVector```.

```scala
val gSc = GaussianScaler(DenseVector(0.0, 1.0, 0.5), DenseVector(1.0, 2.5, 1.5))
//Scale the value of a sample point.
val scVal = gSc(DenseVector(1.0, 1.0, 1.0))
```

**Min Max Feature Scaling**: Implemented in the ```MinMaxScaler``` case class which takes two arguments ```min: DenseVector[Double]``` and ```max: DenseVector[Double]```.

```scala
val mmSc = MinMaxScaler(DenseVector(0.0, 1.0, 0.5), DenseVector(1.0, 2.5, 1.5))
//Scale the value of a sample point.
val scVal = mmSc(DenseVector(1.0, 1.0, 1.0))
```
