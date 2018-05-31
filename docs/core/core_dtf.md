!!! summary
    The `dynaml.tensorflow.dtf` object can be used to create and transform tensors.
    
To use DynaML's tensorflow API, import it in your code/script/DynaML shell session.

```scala
import io.github.mandar2812.dynaml.tensorflow._
import org.platanios.tensorflow.api._
```

## Creating Tensors.

Creating tensors using the `dtf` object is easy, the user needs to provide a scala collection containing the
the data, the shape and data-type of the tensor. 

There is more than one way to instantiate a tensor.

### Enumeration of Values

```scala
import io.github.mandar2812.dynaml.tensorflow._
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
import io.github.mandar2812.dynaml.tensorflow._
import org.platanios.tensorflow.api._

val float_seq = Seq(1f, 2f, 3f, 4f)
val double_seq = Seq(1.0, 2.0, 3.0, 4.0)

//Specify data type as a string, and enumerate the shape
val tensor_float = dtf.tensor_from[Float](
  "FLOAT32", 2, 2)(float_seq)

//Prints out a summary of the values in tensor1
tensor_float.summarize()

val tensor_double = dtf.tensor_from[Double](
  "FLOAT64", 2, 2)(double_seq)

tensor_double.summarize()
```

### From an Array of Bytes.

When dealing with binary data formats, such as images and other binary numerical formats, 
it is useful to be able to instantiate tensors from buffers of raw bytes.

```scala
import io.github.mandar2812.dynaml.tensorflow._
import org.platanios.tensorflow.api._

val byte_buffer: Array[Byte] = _

val shape: Shape = _

val byte_tensor = dtf.tensor_from_buffer(INT32, shape)(byte_buffer)
```

