---
title: DynaML Pipes
---


!!! summary
    In this section we attempt to give a simple yet effective introduction to the data pipes module of DynaML.

## Motivation

Machine Learning involves operations that can be thought of as being part of different stages.

 1. Data pre-processing

    Data _munging_ or pre-processing is one of the most time consuming activities in the analysis   and modeling cycle, yet very few libraries do justice to this need.

 2. Modeling: train, validation & test :

    Training and testing models on data is a cyclical process and in the interest of keeping things manageable, it is important to separate operations in stage 1 from stage 2 and 3.

 3. Post processing: produce and summarize results via reports, visualizations etc.


## What are Data Pipes?

At their heart data pipes in DynaML are (wrapped) Scala functions. Every machine learning workflow can be thought of as a chain of functional transformations on data. These functional transformations are applied one after another (in fancy language _composed_) to yield a result which is then suitable for modeling/training.


### Creating a Pipe

As we mentioned earlier a DynaML pipe is nothing but a thin wrapper around a scala function. Creating a new data pipe is very easy, you just create a scala function and give it to the ``#!scala DataPipe()` object.

```scala

val func = (x: Double) => math.sin(2.0*x)*math.exp(-2.0*x)

val tPipe = DataPipe(func)

```

### Stacking/Composing Data Pipes



```scala
val pre_processing = DataPipe((x: Double) => math.sin(2.0*x)*math.exp(-2.0*x))
val post_processing = DataPipe((x: Double) => if(x <= 0.2) "Y" else "N")

//Compose the two pipes
//The result will be "Y"
val tPipe = pre_processing > post_processing

tPipe(15.5)

```

!!! tip
    It is possible to create a pipe from any scala type to another, inclucing `#!scala Unit`. For example the statement `#!scala val p = DataPipe(() => scala.Random.nextGaussian())` creates a pipe which when executed samples from a univariate gaussian distribution `#!scala val sample = p.run()`

!!! warning ""
    You can compose or stack any number of pipes using the ```>``` character to create a composite data workflow. There is only one constraint when joining two pipes, that the destination type of the first pipe must be the same as the source type of the second pipe, in other words:
    > dont put square pegs into round holes
