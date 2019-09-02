!!! cite ""
    What I cannot create, I do not understand - Richard Feynman

<br/>

!!! summary
    Since version 1.4 a new package called `#!scala probability` has been added to the core api with an aim to aid in the modeling of random variables and measurable functions.

Random variables and probability distributions form the bedrock of modern statistical based approaches to inference. Furthermore, analytically tractable inference is only possible for a small number of models while a wealth of interesting model structures don't yield themselves to analytical inference and approximate sampling based approaches are often employed.

## Random Variable API

Although both random variable with tractable and intractable distributions can be constructed, the emphasis is on the sampling capabilities of random variable objects.

The `#!scala probability` package class hierarchy consists of classes and traits which represent continuous and discrete random variables along with ability to endow them with distributions.

### DynaML Random Variable

The [`#!scala RandomVariable[Domain]`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/index.html#io.github.mandar2812.dynaml.probability.RandomVariable) forms the top of the class hierarchy in the `#!scala probability` package. It is a light weight trait which takes a form like so.

```scala
abstract class RandomVariable[Domain] {

  val sample: DataPipe[Unit, Domain]

  def :*[Domain1](other: RandomVariable[Domain1]): RandomVariable[(Domain, Domain1)] = {
    val sam = this.sample
    RandomVariable(BifurcationPipe(sam,other.sample))
  }
}

```

A `#!scala RandomVariable` instance is defined by its type parameter `#!scala Domain`, in Mathematics this is the underlying space (referred to as the _support_) over which the random variable is defined ($\mathbb{R}^p$ for continuos variables, $\mathbb{N}$ for discrete variables).

The two main functionalities are as follows.

* `#!scala sample` which is a data pipe having no input and outputs a sample from the random variables distribution whenever invoked.

* ```:*``` the 'composition' operator between random variables, evaluating an expression like `#!scala randomVar1 :* randomVar2` creates a new random variable whose domain is a cartesian product of the domains of `#!scala randomVar1` and `#!scala randomVar2`.


Continuous and discrete distribution random variables are implemented through the [`#!scala ContinuousDistrRV[Domain]`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/index.html#io.github.mandar2812.dynaml.probability.ContinuousDistrRV) and [`#!scala DiscreteDistrRV[Domain]`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/index.html#io.github.mandar2812.dynaml.probability.DiscreteDistrRV) respectively.

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

```

!!! note ""
    Sampling is the core functionality of the classes extending `#!scala RandomVariable` but in some cases representing random variables having an underlying (tractable and known) distribution is a requirement, for that purpose there exists the  [`#!scala RandomVarWithDistr[Domain, Dist]`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/index.html#io.github.mandar2812.dynaml.probability.RandomVarWithDistr) trait which is a bare bones extension of `#!scala RandomVariable`; it contains only one other member, `#!scala underlyingDist` which is of abstract type `#!scala Dist`.

    The type `#!scala Dist` can be any breeze distribution, which is either contained in the package `#!scala breeze.stats.distributions` or a user written extension of a breeze probability distribution.


!!! tip "Creating random variables from breeze distributions"
    Creating a random variable backed by a breeze distribution is easy, simply pass the breeze distribution to the `#!scala RandomVariable` companion object.

    ```scala
    val p = RandomVariable(new Beta(7.5, 7.5))
    ```

    The `#!scala RandomVariable` object recognizes the breeze distribution passed to it and creates a continuous or discrete random variable accordingly.
