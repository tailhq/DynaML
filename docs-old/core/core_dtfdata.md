!!! summary
    The `DataSet` API added in v1.5.3, makes it easy to work with potentially large data sets, 
    perform complex pre-processing tasks and feed these data sets into TensorFlow models.
    
    
## Data Set

### Basics

A `DataSet[X]` instance is simply a wrapper over an `Iterable[X]` object, although the user still has 
access to the underlying collection.    
        
!!! tip    
    The [`dtfdata`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/#io.github.mandar2812.dynaml.tensorflow.package) 
    object gives the user easy access to the `DataSet` API.
    
    ```scala
    import _root_.io.github.mandar2812.dynaml.probability._
    import _root_.io.github.mandar2812.dynaml.pipes._
    import io.github.mandar2812.dynaml.tensorflow._
     
     
    val random_numbers = GaussianRV(0.0, 1.0) :* GaussianRV(1.0, 2.0) 
     
    //Create a data set.
    val dataset1 = dtfdata.dataset(random_numbers.iid(10000).draw)
    
    //Access underlying data
    dataset1.data
    ```
    
### Transformations

DynaML data sets support several operations of the _map-reduce_ philosophy.

#### Map

Transform each element of type `X` into some other element of type `Y` (`Y` can possibly be the same as `X`).

```scala
import _root_.io.github.mandar2812.dynaml.probability._
import _root_.io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.tensorflow._
     
     
val random_numbers = GaussianRV(0.0, 1.0)
//A data set of random gaussian numbers.     
val random_gaussian_dataset = dtfdata.dataset(
  random_numbers.iid(10000).draw
)

//Transform data set by applying a scala function
val random_chisq_dataset = random_gaussian_dataset.map((x: Double) => x*x)

val exp_tr = DataPipe[Double, Double](math.exp _)
//Can pass a DataPipe instead of a function
val random_log_gaussian_dataset = random_gaussian_dataset.map(exp_tr)
```

#### Flat Map

Process each element by applying a function which transforms each element into an `Iterable`, 
this operation is followed by flattening of the top level `Iterable`.

Schematically, this process is

`Iterable[X] -> Iterable[Iterable[Y]] -> Iterable[Y]`

```scala
import _root_.io.github.mandar2812.dynaml.probability._
import _root_.io.github.mandar2812.dynaml.pipes._
import scala.util.Random
import io.github.mandar2812.dynaml.tensorflow._

val random_gaussian_dataset = dtfdata.dataset(
  GaussianRV(0.0, 1.0).iid(10000).draw
)

//Transform data set by applying a scala function
val gaussian_mixture = random_gaussian_dataset.flatMap(
  (x: Double) => GaussianRV(0.0, x*x).iid(10).draw
)
```

#### Filter

Collect only the elements which satisfy some predicate, i.e. a function which returns `true` for the
elements to be selected (filtered) and `false` for the ones which should be discarded.

```scala
import _root_.io.github.mandar2812.dynaml.probability._
import _root_.io.github.mandar2812.dynaml.pipes._
import scala.util.Random
import io.github.mandar2812.dynaml.tensorflow._

val gaussian_dataset = dtfdata.dataset(
  GaussianRV(0.0, 1.0).iid(10000).draw
)

val onlyPositive = DataPipe[Double, Boolean](_ > 0.0)

val truncated_gaussian = gaussian_dataset.filter(onlyPositive)

val zeroOrGreater = (x: Double) => x >= 0.0
//filterNot works in the opposite manner to filter
val neg_truncated_gaussian = gaussian_dataset.filterNot(zeroOrGreater)

```

#### Scan & Friends

Sometimes, we need to perform operations on a data set which are sequential in nature. In this situation, 
the `scanLeft()` and `scanRight()` are useful.

Lets simulate a random walk, we start with $x_0$, a number and add independent gaussian increments to it.

$$
\begin{align*}
x_t &= x_{t-1} + \epsilon \\
\epsilon &\sim \mathcal{N}(0, 1)
\end{align*}
$$

```scala
import _root_.io.github.mandar2812.dynaml.probability._
import _root_.io.github.mandar2812.dynaml.pipes._
import scala.util.Random
import io.github.mandar2812.dynaml.tensorflow._

val gaussian_increments = dtfdata.dataset(
  GaussianRV(0.0, 1.0).iid(10000).draw
)

val increment = DataPipe2[Double, Double, Double]((x, i) => x + i)

//Start the random walk from zero, and keep adding increments.
val random_walk = gaussian_increments.scanLeft(0.0)(increment)
```

The `scanRight()` works just like the `scanLeft()` method, except it begins from the last element 
of the collection.

#### Reduce & Reduce Left

The `reduce()` and `reduceLeft()` methods help in computing summary values from the entire data 
collection.

```scala
import _root_.io.github.mandar2812.dynaml.probability._
import _root_.io.github.mandar2812.dynaml.pipes._
import scala.util.Random
import io.github.mandar2812.dynaml.tensorflow._

val gaussian_increments = dtfdata.dataset(
  GaussianRV(0.0, 1.0).iid(10000).draw
)

val increment = DataPipe2[Double, Double, Double]((x, i) => x + i)

val random_walk = gaussian_increments.scanLeft(0.0)(increment)

val average = random_walk.reduce(
  DataPipe2[Double, Double, Double]((x, y) => x + y)
)/10000.0
```

#### Other Transformations

Some times transformations on data sets cannot be applied on each element individually, but the 
entire data collection is required for such a transformation.

```scala
import _root_.io.github.mandar2812.dynaml.probability._
import _root_.io.github.mandar2812.dynaml.pipes._
import scala.util.Random
import io.github.mandar2812.dynaml.tensorflow._

val gaussian_data = dtfdata.dataset(
  GaussianRV(0.0, 1.0).iid(10000).draw
)

val resample = DataPipe[Iterable[Double], Iterable[Double]](
  coll => (0 until 10000).map(_ => coll(Random.nextInt(10000)))
)

val resampled_data = gaussian_data.transform(resample)

```

!!! note 
    **Conversion to TF-Scala `Dataset` class**
  
    The TensorFlow scala API also has a `Dataset` class, from a DynaML `DataSet` 
    instance, it is possible to obtain a TensorFlow `Dataset`.
    
    ```scala
    import _root_.io.github.mandar2812.dynaml.probability._
    import _root_.io.github.mandar2812.dynaml.pipes._
    import io.github.mandar2812.dynaml.tensorflow._
    import org.platanios.tensorflow.api._
    import org.platanios.tensorflow.api.types._
     
     
    val random_numbers = GaussianRV(0.0, 1.0)
     
    //Create a data set.
    val dataset1 = dtfdata.dataset(random_numbers.iid(10000).draw)
    
    //Convert to TensorFlow data set
    dataset1.build[Tensor, Output, DataType.Aux[Double], DataType, Shape](
      Left(DataPipe[Double, Tensor](x => dtf.tensor_f64(1)(x))),
      FLOAT64, Shape(1)    
    )
    ```



## Tuple Data & Supervised Data

The classes `ZipDataSet[X, Y]` and `SupervisedDataSet[X, Y]` both represent data collections which consist of 
`(X, Y)` tuples. They can be created in a number of ways.

### Zip Data

The `zip()` method can be used to create data sets consisting of tuples.

```scala
import _root_.io.github.mandar2812.dynaml.probability._
import _root_.io.github.mandar2812.dynaml.pipes._
import scala.util.Random
import _root_.breeze.stats.distributions._
import io.github.mandar2812.dynaml.tensorflow._

val gaussian_data = dtfdata.dataset(
  GaussianRV(0.0, 1.0).iid(10000).draw
)

val log_normal_data = gaussian_data.map((x: Double) => math.exp(x))

val poisson_data  = dtfdata.dataset(
  RandomVariable(Poisson(2.5)).iid(10000).draw
) 

val tuple_data1 = poisson_data.zip(gaussian_data)

val tuple_data2 = poisson_data.zip(log_normal_data)

//Join on the keys, in this case the 
//Poisson distributed integers

tuple_data1.join(tuple_data2)
```

### Supervised Data

For supervised learning operations, we can use the `SupervisedDataSet` class, which can be instantiated 
in the following ways.

```scala

import _root_.io.github.mandar2812.dynaml.probability._
import _root_.io.github.mandar2812.dynaml.pipes._
import scala.util.Random
import _root_.breeze.stats.distributions._
import io.github.mandar2812.dynaml.tensorflow._

val gaussian_data = dtfdata.dataset(
  GaussianRV(0.0, 1.0).iid(10000).draw
)

val sup_data1 = gaussian_data.to_supervised(
  DataPipe[Double, (Double, Double)](x => (x, GaussianRV(0.0, x*x).draw))
)

val targets = gaussian_data.map((x: Double) => math.exp(x))

val sup_data2 = dtfdata.supervised_dataset(gaussian_data, targets)

```

