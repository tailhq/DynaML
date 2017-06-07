!!! summary
    The pipes API provides a good foundation to construct data processing pipelines, in this section we show how it is extended for application to a specific application i.e. attribute scaling & transformation.  

Transforming data attributes is an often repeated task, some examples include re-scaling values in a finite domain $[min, max]$, gaussian centering, _principal component analysis_ (PCA), _discreet Haar wavelet_ (DWT) transform etc.

The pipes API contains traits for these tasks, they are abstract skeletons which can be extended by the user to create arbitrary feature re-scaling transformations.

## Encoders

`#!scala Encoder[I, J]` are an extension of `#!scala DataPipe[I, J]` class which has an extra value member `#!scala i: DataPipe[J, I]` which represents the inverse transformation.

!!! note
    `#!scala Encoder[I, J]` implies a reversible, one to one transformation of the input. Mathematically this can be expressed as

    $$
    \begin{align}
    g: \mathcal{X} &\rightarrow \mathcal{Y} \\
    h: \mathcal{Y} &\rightarrow \mathcal{X} \\
    h(g(x)) &= x \ \ \ \forall x \in \mathcal{X} \\
    h &\equiv g^{-1}
    \end{align}
    $$

An encoder can be created by calling the `apply` method of the `#!scala Encoder` object.

```scala
//Converts a point expressed in cartesian coordinates
//into a point expressed in polar coordinates and vice versa.
val cartesianToPolar = Encoder(
  (pointCart: (Double, Double)) => {
    val (x,y) = pointCart
    val r = math.sqrt(x*x + y*y)
    if(r != 0.0) (r, math.arcsin(y/r)) else (0.0, 0.0)
  }),
  (pointPolar: (Double, Double)) => {
    val (r, theta) = pointPolar
    (r*math.cos(theta), r*math.sin(theta))
  }
)

```

!!! note
    In the above example, we created a cartesian to polar coordinate converter by specifying the forward and reverse transformations as anonymous scala functions. But we could as well have passed the forward and reverse transforms as `#!scala DataPipe` instances.

    ```scala
    val forwardTransform: DataPipe[I, J] = _
    val reverseTransform: DataPipe[J, I] = _
    //Still works.
    val enc = Encoder(forwardTransform, reverseTransform)
    ```

## Scalers

`#!scala Scaler[I]` is an extension of the `#!scala DataPipe[I, I]` trait. Represents transformations of inputs which don't change their type.

```scala

val linTr = Scaler((x: Double) => x*5.0 + -1.5)

```

## Reversible Scalers

`#!scala ReversibleScaler[I]` extends `#!scala Scaler[I]` along with `#!scala Encoder[I, J]`, a reversible re-scaling of inputs.

!!! note "The `#!scala >` and `#!scala *` for scalers and encoders"

    Since `#!scala Encoder[S, D]`, `#!scala Scaler[S]` and `#!scala ReversibleScaler[S, D]` are inherit the `#!scala DataPipe` trait, they can be composed with any data pipeline as usual, but there are special cases.

    If an `#!scala Encoder[I, J]` instance is composed with `#!scala Encoder[J, K]`, the result is of type `#!scala Encoder[I, K]` and accordingly for `#!scala Scaler[I]` and `#!scala ReversibleScaler[I]`.

    The `*` can be used to create cartesian products of encoders and scalers.

    ```scala
    val enc1: Encoder[I, J] = _
    val enc2: Encoder[K, L] = _
    val enc3: Encoder[(I, K), (J, L)] = enc1*enc2
    ```





!!! tip
    Common attribute transformations like gaussian centering, min-max scaling, etc are included in the `#!scala dynaml.utils` package, click [here](/utils/scalers.md) to see their syntax.
