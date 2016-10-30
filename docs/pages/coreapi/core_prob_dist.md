---
title: Random Variables with Distributions
tags: [probability]
sidebar: coreapi_sidebar
permalink: core_prob_dist.html
folder: coreapi
---

Sampling is the core functionality of the classes extending ```RandomVariable``` but in some cases representing random variables having an underlying (tractable and known) distribution is a requirement, for that purpose there exists the  [```RandomVarWithDistr[Domain, Dist]```]({{site.apiurl}}/dynaml-core/index.html#io.github.mandar2812.dynaml.probability.RandomVarWithDistr) trait which is a bare bones extension of ```RandomVariable```; it contains only one other member, ```underlyingDist``` which is of abstract type ```Dist```.

The type ```Dist``` is any breeze distribution, which is either contained in the package ```breeze.stats.distributions``` or a user written extension of a breeze probability distribution.

Creating a random variable backed by a breeze distribution is easy, simply pass the breeze distribution to the ```RandomVariable``` companion object.

```scala
val p = RandomVariable(new Beta(7.5, 7.5))
```

Continuous and discrete distribution random variables are implemented through the [```ContinuousDistrRV[Domain]```]({{site.apiurl}}/dynaml-core/index.html#io.github.mandar2812.dynaml.probability.ContinuousDistrRV) and [```DiscreteDistrRV[Domain]```]({{site.apiurl}}/dynaml-core/index.html#io.github.mandar2812.dynaml.probability.DiscreteDistrRV) respectively. The ```RandomVariable``` object recognizes the breeze distribution passed to it and creates a continuous or discrete random variable accordingly.
