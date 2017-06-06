---
title: Data Pipes Hierarchy
---

## Base

At the top of the pipes hierarchy is the base trait [```DataPipe[Source, Destination]```](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-pipes/index.html#io.github.mandar2812.dynaml.pipes.DataPipe) which is a thin wrapper for a Scala function having the type ```(Source) => Destination```. Along with that the base trait also defines how pipes are composed with each other to yield more complex workflows.

### Pipes in Parallel

The [```ParallelPipe[Source1, Result1, Source2, Result2]```](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-pipes/index.html#io.github.mandar2812.dynaml.pipes.ParallelPipe) trait models pipes which are attached to each other, from an implementation point of view these can be seen as data pipes taking input from ```(Source1, Source2)``` and yielding values from ```(Result1, Result2)```. They can be created in two ways:

By supplying two pipes to the ```DataPipe()``` object.

```scala
val pipe1 = DataPipe((x: Double) => math.sin(2.0*x)*math.exp(-2.0*x))
val pipe2 = DataPipe((x: Double) => if(x <= 0.2) "Y" else "N")

val pipe3 = DataPipe(pipe1, pipe2)
//Returns (-0.013, "N")
pipe3((2.0, 15.0))

```

By duplicating a single pipe using [```DynaMLPipe.duplicate```](/pipes/pipes_misc.md#duplicate-a-pipe)

```scala
//Already imported in DynaML repl
//but should be imported when using DynaML API
//outside of its provided repl environment.
import io.github.mandar2812.dynaml.DynaMLPipe._

val pipe1 = DataPipe((x: Double) => math.sin(2.0*x)*math.exp(-2.0*x))

val pipe3 = duplicate(pipe1)
//Returns (-0.013, -9E-14)
pipe3((2.0, 15.0))

```


### Diverging Pipes

The [```BifurcationPipe[Source, Result1, Result2]```](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-pipes/index.html#io.github.mandar2812.dynaml.pipes.BifurcationPipe) trait represents pipes which start from the same source and yield two result types, from an implementation point of view these can be seen as data pipes taking input from ```Source1``` and yielding values from ```(Result1, Result2)```. They can be created in two ways:

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

In order to enable pipes which have side effects i.e. writing to disk, the [```SideEffectPipe[Source]```](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-pipes/index.html#io.github.mandar2812.dynaml.pipes.SideEffectPipe) trait is used. Conceptually it is a pipe taking as input a value from ```Source``` but has a return type of ```Unit```.

## Stream Processing

To simplify writing pipes for scala streams, the [```StreamDataPipe[I, J, K]```](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-pipes/index.html#io.github.mandar2812.dynaml.pipes.StreamDataPipe) and its subclasses implement workflows on streams.  

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

****************************

The following API members were added in [v1.4.1](https://github.com/transcendent-ai-labs/DynaML/releases/tag/v1.4.1)

### Flat Map

`#!scala StreamFlatMapPipe` carries out the scala flat-map operation on a stream.

```scala
val mapFunc = (n: Int) => (1 to n).sliding(2).toStream
val streamFMPipe = StreamFlatMapPipe(mapFunc)

streamFMPipe((1 to 20).toStream)
```

!!! tip "Pipes on Spark RDDs"

    It is also possible to create pipes acting on Spark RDDs.

    ```scala
    val num = 20
    val sc: SparkContext = _
    val numbers = sc.parallelize(1 to num)
    val convPipe = RDDPipe((n: Int) => n.toDouble)

    val sqPipe = RDDPipe((x: Double) => x*x)

    val sqrtPipe = RDDPipe((x: Double) => math.sqrt(x))

    val resultPipe = RDDPipe((r: RDD[Double]) => r.reduce(_+_).toInt)

    val netPipeline = convPipe > sqPipe > sqrtPipe > resultPipe
    netPipeline(numbers)
    ```

## Advanced Pipes

Apart from the basic capabilities offered by the `#!scala DataPipe[Source, Destination]` interface and its family, users can also work with
more complex workflow components some of which are shown below.

The advanced components of the pipes API enable two key extensions.

 - Data pipes which take more than one argument[^1].
 - Data pipes which take an argument and return a data pipe[^2]

### Data Pipe 2

`#!scala DataPipe2[A, B, C]`

*arguments*: 2 of type `A` and `B` respectively

*returns*: result of type `C`

```scala

val f2: (A, B) => C = _  
val pipe2 = DataPipe2(f2)

```

### DataPipe 3

`#!scala DataPipe3[A, B, C, D]`

*arguments*: 3 of type `A`, `B` and `C` respectively

*returns*: result of type `D`

```scala

val f3: (A, B, C) => D = _  
val pipe3 = DataPipe3(f3)

```


### DataPipe 4

`#!scala DataPipe4[A, B, C, D, E]`

*arguments*: 4 of type `A`, `B`, `C` and `D` respectively

*returns*: result of type `E`

```scala

val f4: (A, B, C, D) => E = _  
val pipe4 = DataPipe4(f4)

```


[^1]: so far DynaML has support till 4
[^2]: similar to curried functions in scala.



### Meta Pipe

`#!scala MetaPipe[A, B, C]`

Takes an argument returns a `DataPipe`

### Meta Pipe (2, 1)

`#!scala MetaPipe21[A, B, C, D]`

Takes 2 arguments returns a `DataPipe`

### Meta Pipe (1, 2)

`#!scala MetaPipe12[A, B, C, D]`

Takes an argument returns a `DataPipe2`
