---
title: DynaML Pipes
keywords: mydoc
tags: [pipes, workflow]
sidebar: product1_sidebar
permalink: p1_pipes.html
folder: product1
---


Data _munging_ or pre-processing is one of the most time consuming activities in the analysis and modeling cycle, yet very few libraries do justice to this need. In DynaML the aim has been to make data analysis more reproducible and easy, hence designing, maintaining and improving a powerful data workflow framework is at the center of the development endeavour. In this section we attempt to give a simple yet effective introduction to the data pipes module of DynaML.


## What are DynaML Data Pipes?

At their heart data pipes in DynaML are (thinly wrapped) Scala functions. Every pre-processing workflow can be visualized as a chain of functional transformations on the data. These functional transformations are applied one after another (in fancy language _composed_) to yield a result which is then suitable for modeling/training.


### Creating an arbitrary pipe

As we mentioned earlier a DynaML pipe is nothing but a thin wrapper around a scala function. Creating a new data pipe is very easy, you just create a scala function and give it to the ```DataPipe()``` object.

```scala

val func = (x: Double) => math.sin(2.0*x)*math.exp(-2.0*x)

val tPipe = DataPipe(func)

```

### Stacking/Composing Data Pipes

You can compose or stack any number of pipes using the ```>``` character to create a composite data workflow. There is only one constraint when joining two pipes, that the destination type of the first pipe must be the same as the source type of the second pipe, in other words:

>dont put square pegs into round holes

```scala
val pre_processing = DataPipe((x: Double) => math.sin(2.0*x)*math.exp(-2.0*x))
val post_processing = DataPipe((x: Double) => if(x <= 0.2) "Y" else "N")

//Compose the two pipes
//The result will be "Y"
val tPipe = pre_processing > post_processing

tPipe(15.5)

```

{% include links.html %}
