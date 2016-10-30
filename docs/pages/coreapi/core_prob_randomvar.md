---
title: Random Variables
sidebar: coreapi_sidebar
tags: [probability]
permalink: core_prob_randomvar.html
folder: coreapi
---

>What I cannot create, I do not understand.
> - Richard Feynman

Random variables and probability distributions form the bedrock of modern statistical based approaches to inference. Furthermore, analytically tractable inference is only possible for a small number of models while a wealth of interesting model structures don't yield themselves to analytical inference and approximate sampling based approaches are often employed.

## Random Variable API

Since version 1.4 a new package called ```probability``` has been added to the core api with an aim to aid in the modeling of random variables and measurable functions. Although both random variable with tractable and intractable distributions can be constructed, the emphasis is on the sampling capabilities of random variable objects.

The ```probability``` package class hierarchy consists of classes and traits which represent continuous and discrete random variables along with ability to endow them with distributions.

### DynaML Random Variable

The [```RandomVariable[Domain]```]({{site.apiurl}}/dynaml-core/index.html#io.github.mandar2812.dynaml.probability.RandomVariable) forms the top of the class hierarchy in the ```probability``` package. It is a light weight trait which takes a form like so.

```scala
abstract class RandomVariable[Domain] {

  val sample: DataPipe[Unit, Domain]

  def :*[Domain1](other: RandomVariable[Domain1]): RandomVariable[(Domain, Domain1)] = {
    val sam = this.sample
    RandomVariable(BifurcationPipe(sam,other.sample))
  }
}

```

A ```RandomVariable``` instance is defined by its type parameter ```Domain```, in Mathematics this is the underlying space (referred to as the _support_) over which the random variable is defined ($$\mathbb{R}^p$$ for continuos variables, $$\mathbb{N}$$ for discrete variables).

The two main functionalities are as follows.

* ```sample``` which is a data pipe having no input and outputs a sample from the random variables distribution whenever invoked.

* ```:*``` the 'composition' operator between random variables, evaluating an expression like ```randomVar1 :* randomVar2``` creates a new random variable whose domain is a cartesian product of the domains of ```randomVar1``` and ```randomVar2```.

### Creating Random Variables

Creating random variables can be created by a number of ways.

```scala
import breeze.stats.distributions._
import spire.implicits._


//Create a sampling function
val sampF: () => Double = ...
val rv = RandomVariable(sampF)

//Also works with a pipe
val sampF: DataPipe[Unit, Double] = ...
val rv = RandomVariable(sampF)

//Using a breeze distribution
val p = RandomVariable(new Beta(7.5, 7.5))

```
