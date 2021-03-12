!!! summary
    The [`dtf`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/#io.github.mandar2812.dynaml.tensorflow.package) 
    object can be used to create and transform tensors.
    
To use DynaML's tensorflow API, import it in your code/script/DynaML shell session.

```scala
import io.github.tailhq.dynaml.tensorflow._
import org.platanios.tensorflow.api._
```

## Creating Tensors.

Creating tensors using the [`dtf`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/#io.github.mandar2812.dynaml.tensorflow.package) 
object is easy, the user needs to provide a scala collection containing the
the data, the shape and data-type of the tensor. 

There is more than one way to instantiate a tensor.

### Enumeration of Values

```scala
import io.github.tailhq.dynaml.tensorflow._
import org.platanios.tensorflow.api._

//Create a float tensor
val tensor_float = dtf.tensor_from[Float](
  FLOAT32, Shape(2, 2))(
  1f, 2f, 3f, 4f)

//Prints out a summary of the values in tensor1
tensor_float.summarize()

val tensor_double = dtf.tensor_from[Double](
  FLOAT64, Shape(2, 2))(
  1.0, 2.0, 3.0, 4.0)

tensor_double.summarize()
```

### From a Scala Sequence

```scala
import io.github.tailhq.dynaml.tensorflow._
import org.platanios.tensorflow.api._

val float_seq = Seq(1f, 2f, 3f, 4f)
val double_seq = Seq(1.0, 2.0, 3.0, 4.0)

//Specify data type as a string, and enumerate the shape
val tensor_float = dtf.tensor_from[Float]("FLOAT32", 2, 2)(float_seq)

//Prints out a summary of the values in tensor1
tensor_float.summarize()

val tensor_double = dtf.tensor_from[Double]("FLOAT64", 2, 2)(double_seq)

tensor_double.summarize()
```

### From an Array of Bytes.

When dealing with binary data formats, such as images and other binary numerical formats, 
it is useful to be able to instantiate tensors from buffers of raw bytes.

```scala
import io.github.tailhq.dynaml.tensorflow._
import org.platanios.tensorflow.api._

val byte_buffer: Array[Byte] = _

val shape: Shape = _

val byte_tensor = dtf.tensor_from_buffer(INT32, shape)(byte_buffer)
```

Apart from these functions, there are.

```scala
import io.github.tailhq.dynaml.tensorflow._
import org.platanios.tensorflow.api._

//Double tensor
val t = dtf.tensor_f64(2, 2)(1.0, 2.0, 3.0, 4.0)

//32 bit Integer tensor
val t_int = dtf.tensor_i32(2, 3)(1, 2, 3, 4, 5, 6)

//Fill a (3, 2, 5) tensor, with the value 1.
val t_fill = dtf.fill(FLOAT32, 3, 2, 5)(1f)

```

### Random Tensors

It is also possible to create tensors whose elements are _independent and identically distributed_, by using the DynaML
probability API.

```scala
import breeze.stats.distributions._

import io.github.tailhq.dynaml.probability._
import io.github.tailhq.dynaml.tensorflow._
import org.platanios.tensorflow.api._

val rv = RandomVariable(new LogNormal(0.0, 1.5))

val random_tensor = dtf.random(FLOAT64, 3, 5, 2)(rv)

```

## Operations on Tensors

### Stack

```scala
DynaML>val random_tensor1 = dtf.random(FLOAT64, 2, 3)(rv) 
random_tensor1: Tensor = FLOAT64[2, 3]

DynaML>val random_tensor2 = dtf.random(FLOAT64, 2, 3)(rv) 
random_tensor2: Tensor = FLOAT64[2, 3]

DynaML>val t = dtf.stack(Seq(random_tensor1, random_tensor2), axis = 1) 
t: Tensor = FLOAT64[2, 2, 3]

DynaML>val t0 = dtf.stack(Seq(random_tensor1, random_tensor2), axis = 0) 
t0: Tensor = FLOAT64[2, 2, 3]

DynaML>random_tensor1.summarize(100, false) 
res18: String = """FLOAT64[2, 3]
[[0.3501699906342581, 0.2900664662305818, 0.42806656451314345],
 [0.3066005571688877, 1.3931959054429162, 0.6366232162759474]]"""

DynaML>random_tensor2.summarize(100, false) 
res19: String = """FLOAT64[2, 3]
[[0.21565105620570899, 0.5267519630011802, 6.817248106561024],
 [0.35121879449734744, 5.487926862392467, 2.3538094624119177]]"""

DynaML>t.summarize(100, false) 
res16: String = """FLOAT64[2, 2, 3]
[[[0.3501699906342581, 0.2900664662305818, 0.42806656451314345],
  [0.21565105620570899, 0.5267519630011802, 6.817248106561024]],

 [[0.3066005571688877, 1.3931959054429162, 0.6366232162759474],
  [0.35121879449734744, 5.487926862392467, 2.3538094624119177]]]"""

DynaML>t0.summarize(100, false) 
res17: String = """FLOAT64[2, 2, 3]
[[[0.3501699906342581, 0.2900664662305818, 0.42806656451314345],
  [0.3066005571688877, 1.3931959054429162, 0.6366232162759474]],

 [[0.21565105620570899, 0.5267519630011802, 6.817248106561024],
  [0.35121879449734744, 5.487926862392467, 2.3538094624119177]]]"""


```

### Concatenate

```scala

DynaML>val t = dtf.concatenate(Seq(random_tensor1, random_tensor2), axis = 0) 
t: Tensor = FLOAT64[4, 3]

DynaML>val t1 = dtf.concatenate(Seq(random_tensor1, random_tensor2), axis = 1) 
t1: Tensor = FLOAT64[2, 6]

DynaML>t.summarize(100, false) 
res28: String = """FLOAT64[4, 3]
[[0.3501699906342581, 0.2900664662305818, 0.42806656451314345],
 [0.3066005571688877, 1.3931959054429162, 0.6366232162759474],
 [0.21565105620570899, 0.5267519630011802, 6.817248106561024],
 [0.35121879449734744, 5.487926862392467, 2.3538094624119177]]"""

DynaML>t1.summarize(100, false) 
res29: String = """FLOAT64[2, 6]
[[0.3501699906342581, 0.2900664662305818, 0.42806656451314345, 0.21565105620570899, 0.5267519630011802, 6.817248106561024],
 [0.3066005571688877, 1.3931959054429162, 0.6366232162759474, 0.35121879449734744, 5.487926862392467, 2.3538094624119177]]"""


```

### Unstack

```scala
DynaML>dtf.unstack(t1, axis = 1) 
res31: Seq[Tensor] = ArraySeq(FLOAT64[2], FLOAT64[2], FLOAT64[2], FLOAT64[2], FLOAT64[2], FLOAT64[2])

DynaML>res31.map(t => t.summarize(100, false)) 
res33: Seq[String] = ArraySeq(
  """FLOAT64[2]
[0.3501699906342581, 0.3066005571688877]""",
  """FLOAT64[2]
[0.2900664662305818, 1.3931959054429162]""",
  """FLOAT64[2]
[0.42806656451314345, 0.6366232162759474]""",
  """FLOAT64[2]
[0.21565105620570899, 0.35121879449734744]""",
  """FLOAT64[2]
[0.5267519630011802, 5.487926862392467]""",
  """FLOAT64[2]
[6.817248106561024, 2.3538094624119177]"""
)


```