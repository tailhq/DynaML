!!! summary ""
    Version 1.5.1 of DynaML, released September 20, 2017, introduces bug fixes and some useful new features
    
## Additions

**Package** `dynaml.probability.distributions`

 - Added [Kumaraswamy](https://en.wikipedia.org/wiki/Kumaraswamy_distribution) distribution, an alternative to the Beta distribution.
 - Added [Erlang](https://en.wikipedia.org/wiki/Erlang_distribution) distribution, a special case of the Gamma distribution.

## Bug Fixes

**Package** `dynaml.kernels`

 - Fixed bug concerning hyper-parameter blocking in `CompositeCovariance` and its children.

**Package** `dynaml.probability.distributions`

 - Fixed calculation error for normalisation constant of multivariate T and Gaussian family.