!!! summary
    The pipes API provides a good foundation to construct data processing pipelines, in this section we show how it is extended for application to a specific application i.e. attribute scaling & transformation.  

Transforming data attributes is an often repeated task, some examples include re-scaling values in a finite domain $[min, max]$, gaussian centering, _principal component analysis_ (PCA), _discreet Haar wavelet_ (DWT) transform etc.

The pipes API contains traits for these tasks, they are abstract skeletons which can be extended by the user to create arbitrary feature re-scaling transformations.

## Encoders

`#!scala Encoder[I, J]` are an extension of `#!scala DataPipe[I, J]` class which has an extra value member `#!scala i: DataPipe[J, I]` which represents the inverse transformation.

!!! tip
    `#!scala Encoder[I, J]` implies a reversible, one to one transformation of the input. Mathematically this can be expressed as

    $$
    \begin{align}
    g: \mathcal{X} &\rightarrow \mathcal{Y} \\
    h: \mathcal{Y} &\rightarrow \mathcal{X} \\
    h(g(x)) &= x \ \ \ \forall x \in \mathcal{X} \\
    h &\equiv g^{-1}
    \end{align}
    $$


## Scalers

`#!scala Scaler[I]` is an extension of the `#!scala DataPipe[I, I]` trait. Represents re-scaling of inputs.

## Reversible Scalers

`#!scala ReversibleScaler[I]` extends `#!scala Scaler[I]` along with `#!scala Encoder[I, J]`, a reversible re-scaling of inputs.

!!! tip
    Common attribute transformations like gaussian centering, min-max scaling, etc are included in the `#!scala dynaml.utils` package, click [here](/utils/scalers.md) to see their syntax.
