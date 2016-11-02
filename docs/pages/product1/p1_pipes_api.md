---
title: Data Pipes Hierarchy
sidebar: product1_sidebar
tags: [pipes, workflow]
permalink: p1_pipes_api.html
folder: product1
---

## Base

At the top of the pipes hierarchy is the base trait [```DataPipe[Source, Destination]```]({{site.apiurl}}/dynaml-pipes/index.html#io.github.mandar2812.dynaml.pipes.DataPipe) which is a thin wrapper for a Scala function having the type ```(Source) => Destination```. Along with that the base trait also defines how pipes are composed with each other to yield more complex workflows.

### Pipes in Parallel

The [```ParallelPipe[Source1, Result1, Source2, Result2]```]({{site.apiurl}}/dynaml-pipes/index.html#io.github.mandar2812.dynaml.pipes.ParallelPipe) trait models pipes which are attached to each other, from an implementation point of view these can be seen as data pipes taking input from ```(Source1, Source2)``` and yielding values from ```(Result1, Result2)```. They can be created in two ways:

By supplying two pipes to the ```DataPipe()``` object.

```scala
val pipe1 = DataPipe((x: Double) => math.sin(2.0*x)*math.exp(-2.0*x))
val pipe2 = DataPipe((x: Double) => if(x <= 0.2) "Y" else "N")

val pipe3 = DataPipe(pipe1, pipe2)
//Returns (-0.013, "N")
pipe3((2.0, 15.0))

```

By duplicating a single pipe using [```DynaMLPipe.duplicate```]({{site.baseurl}}/p1_sample5.html#duplicate-a-pipe)

```scala
val pipe1 = DataPipe((x: Double) => math.sin(2.0*x)*math.exp(-2.0*x))

val pipe3 = DynaMLPipe.duplicate(pipe1)
//Returns (-0.013, -9E-14)
pipe3((2.0, 15.0))

```


### Diverging Pipes

The [```BifurcationPipe[Source, Result1, Result2]```]({{site.apiurl}}/dynaml-pipes/index.html#io.github.mandar2812.dynaml.pipes.BifurcationPipe) trait represents pipes which start from the same source and yield two result types, from an implementation point of view these can be seen as data pipes taking input from ```Source1``` and yielding values from ```(Result1, Result2)```. They can be created in two ways:

By supplying a function of type ```(Source) => (Result1, Result2)``` to the ```DataPipe()``` object.

```scala
val pipe1 = DataPipe((x: Double) => (1.0*math.sin(2.0*x)*math.exp(-2.0*x), math.exp(-2.0*x)))

pipe1(2.0)

```

By using the ```BifurcationPipe()``` object

```scala
val pipe1 = DataPipe((x: Double) => math.sin(2.0*x)*math.exp(-2.0*x))
val pipe2 = DataPipe((x: Double) => if(x <= 0.2) "Y" else "N")

val pipe3 = BifurcationPipe(pipe1, pipe2)
pipe3(2.0)

```


### Side Effects

In order to enable pipes which have side effects i.e. writing to disk, the [```SideEffectPipe[Source]```]({{site.apiurl}}/dynaml-pipes/index.html#io.github.mandar2812.dynaml.pipes.SideEffectPipe) trait is used. Conceptually it is a pipe taking as input a value from ```Source``` but has a return type of ```Unit```.

## Stream Processing

To simplify writing pipes for scala streams, the [```StreamDataPipe[I, J, K]```]({{site.apiurl}}/dynaml-pipes/index.html#io.github.mandar2812.dynaml.pipes.StreamDataPipe) and its subclasses implement workflows on streams.  

### Map

Map every element of a stream.

```scala
val pipe1 = StreamDataPipe((x: Double) => math.sin(2.0*x)*math.exp(-2.0*x))

val str: Stream[Double] = (1 to 5).map(_.toDouble).toStream
pipe1(str)
```

### Filter

Filter certain elements of a stream.

```scala
val pipe1 = StreamDataPipe((x: Double) => x <= 2.5)

val str: Stream[Double] = (1 to 5).map(_.toDouble).toStream
pipe1(str)
```

### Bifurcate stream

```scala
val pipe1 = StreamPartitionPipe((x: Double) => x <= 2.5)

val str: Stream[Double] = (1 to 5).map(_.toDouble).toStream
pipe1(str)
```


### Side effect

```scala
val pipe1 = StreamDataPipe((x: Double) => println("Number is: "+x))

val str: Stream[Double] = (1 to 5).map(_.toDouble).toStream
pipe1(str)
```
