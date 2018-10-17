package io.github.mandar2812.dynaml.tensorflow.api

import java.nio.ByteBuffer

import io.github.mandar2812.dynaml.probability.RandomVariable
import org.platanios.tensorflow.api.tensors.TensorConvertible
import org.platanios.tensorflow.api.types.{DataType, SupportedType}
import org.platanios.tensorflow.api._

private[tensorflow] object Api {

  /**
    * Construct a tensor from a list of elements.
    *
    * @tparam T The type of the elements
    *
    * @param dtype The tensorflow data type of the elements,
    *              this is usually defined by tensorflow scala
    *              i.e. FLOAT64, INT32 etc
    *
    * @param shape The shape of the tensor i.e. Shape(1,2,3)
    *              denotes a rank 3 tensor with 1, 2 and 3 dimensions
    *              for the ranks respectively.
    *
    * @param buffer The elements of type [[T]], can accept varying
    *               number of arguments.
    *
    * @return A tensorflow [[Tensor]] of the appropriate data type
    *         and shape.
    *
    * <b>Usage</b> dtf.tensor_from(INT32, Shape(1, 2, 3))(1, 2, 3, 4, 5, 6)
    *
    * */
  def tensor_from[T, D <: DataType](
    dtype: D, shape: Shape)(buffer: T*)(
    implicit ev: TensorConvertible.Aux[T, D]): Tensor[D] = {
    Tensor(dtype, buffer.head, buffer.tail:_*).reshape(shape)
  }

  /**
    * Construct a tensor from a list of elements.
    *
    * @tparam T The type of the elements
    *
    * @param dtype The tensorflow data type of the elements,
    *              as a string i.e. "FLOAT64", "INT32" etc
    *
    * @param shape The shape of the tensor given as any number
    *              of integer arguments.
    *
    * @param buffer The elements of type [[T]], as a Sequence
    *
    * @return A tensorflow [[Tensor]] of the appropriate data type
    *         and shape.
    *
    * <b>Usage</b> dtf.tensor_from("INT32", 1, 2, 3)((1 to 6).toSeq)
    *
    * */
  def tensor_from[T, D <: DataType](
    dtype: D, shape: Int*)(buffer: Seq[T])(
    implicit ev: TensorConvertible.Aux[T, D]): Tensor[D] = {
    Tensor(dtype, buffer.head, buffer.tail:_*).reshape(Shape(shape:_*))
  }

  /**
    * Construct a tensor from a array of bytes.
    *
    * @tparam T The type of the elements
    *
    * @param dtype The tensorflow data type of the elements,
    *              as a string i.e. "FLOAT64", "INT32" etc
    *
    * @param shape The shape of the tensor given as any number
    *              of integer arguments.
    *
    * @param buffer The elements as a contiguous array of bytes
    *
    * @return A tensorflow [[Tensor]] of the appropriate data type
    *         and shape.
    *
    * <b>Usage</b> dtf.tensor_from_buffer(FLOAT32, 1, 1)((1 to 4).toArray.map(_.toByte))
    *
    * */
  def tensor_from_buffer[T, D <: DataType](
    dtype: D, shape: Shape)(
    buffer: Array[Byte]): Tensor[D] = {
    Tensor.fromBuffer(dtype, shape, buffer.length.toLong, ByteBuffer.wrap(buffer))
  }

  /**
    * Construct a tensor from a array of bytes.
    *
    * @param dtype The tensorflow data type of the elements,
    *              as a string i.e. "FLOAT64", "INT32" etc
    *
    * @param shape The shape of the tensor given as any number
    *              of integer arguments.
    *
    * @param buffer The elements as a contiguous array of bytes
    *
    * @return A tensorflow [[Tensor]] of the appropriate data type
    *         and shape.
    *
    * <b>Usage</b> dtf.tensor_from_buffer("FLOAT32", 1, 1)((1 to 4).toArray.map(_.toByte))
    *
    * */
  def tensor_from_buffer[D <: DataType](
    dtype: D, shape: Int*)(
    buffer: Array[Byte]): Tensor[D] =
    Tensor.fromBuffer[D](
      dtype, Shape(shape:_*),
      buffer.length.toLong, ByteBuffer.wrap(buffer))


  /**
    * Construct an 16 bit integer tensor from a list of elements.
    *
    * @param shape The shape of the tensor given as any number
    *              of integer arguments.
    *
    * @param buffer The elements in row major format.
    *
    * @return A tensorflow [[Tensor]] of the appropriate data type
    *         and shape.
    *
    * <b>Usage</b> dtf.tensor_i16(1, 2, 3)(1, 2, 3, 4, 5, 6)
    *
    * */
  def tensor_i16(shape: Int*)(buffer: Short*)(implicit ev: TensorConvertible.Aux[Short, INT16]): Tensor[INT16] =
    Tensor(INT16, buffer.head, buffer.tail:_*).reshape(Shape(shape:_*))

  /**
    * Construct an 32 bit integer tensor from a list of elements.
    *
    * @param shape The shape of the tensor given as any number
    *              of integer arguments.
    *
    * @param buffer The elements in row major format.
    *
    * @return A tensorflow [[Tensor]] of the appropriate data type
    *         and shape.
    *
    * <b>Usage</b> dtf.tensor_i32(1, 2, 3)(1, 2, 3, 4, 5, 6)
    *
    * */
  def tensor_i32(shape: Int*)(buffer: Int*)(implicit ev: TensorConvertible.Aux[Int, INT32]): Tensor[INT32] =
    Tensor(INT32, buffer.head, buffer.tail:_*).reshape(shape)

  /**
    * Construct an 64 bit integer tensor from a list of elements.
    *
    * @param shape The shape of the tensor given as any number
    *              of integer arguments.
    *
    * @param buffer The elements in row major format.
    *
    * @return A tensorflow [[Tensor]] of the appropriate data type
    *         and shape.
    *
    * <b>Usage</b> dtf.tensor_i64(1, 2, 3)(1, 2, 3, 4, 5, 6)
    *
    * */
  def tensor_i64(shape: Int*)(buffer: Long*)(implicit ev: TensorConvertible.Aux[Long, INT64]): Tensor[INT64] =
    Tensor(INT64, buffer.head, buffer.tail:_*).reshape(Shape(shape:_*))

  /**
    * Construct an 16 bit floating point tensor from a list of elements.
    *
    * @param shape The shape of the tensor given as any number
    *              of integer arguments.
    *
    * @param buffer The elements in row major format.
    *
    * @return A tensorflow [[Tensor]] of the appropriate data type
    *         and shape.
    *
    * <b>Usage</b> dtf.tensor_f16(1, 2, 3)(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    *
    * */
  def tensor_f16(shape: Int*)(buffer: Float*)(implicit ev: TensorConvertible.Aux[Float, FLOAT32]): Tensor[FLOAT16] =
    Tensor(buffer.head, buffer.tail:_*).reshape(shape).cast(FLOAT16)

  /**
    * Construct an 32 bit floating point tensor from a list of elements.
    *
    * @param shape The shape of the tensor given as any number
    *              of integer arguments.
    *
    * @param buffer The elements in row major format.
    *
    * @return A tensorflow [[Tensor]] of the appropriate data type
    *         and shape.
    *
    * <b>Usage</b> dtf.tensor_f32(1, 2, 3)(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    *
    * */
  def tensor_f32(shape: Int*)(buffer: Float*)(implicit ev: TensorConvertible.Aux[Float, FLOAT32]): Tensor[FLOAT32] =
    Tensor(FLOAT32, buffer.head, buffer.tail:_*).reshape(shape)

  /**
    * Construct an 64 bit floating point tensor from a list of elements.
    *
    * @param shape The shape of the tensor given as any number
    *              of integer arguments.
    *
    * @param buffer The elements in row major format.
    *
    * @return A tensorflow [[Tensor]] of the appropriate data type
    *         and shape.
    *
    * <b>Usage</b>  dtf.tensor_f64(1, 2, 3)(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    *
    * */
  def tensor_f64(shape: Int*)(buffer: Double*)(implicit ev: TensorConvertible.Aux[Double, FLOAT64]): Tensor[FLOAT64] =
    Tensor(FLOAT64, buffer.head, buffer.tail:_*).reshape(shape)


  /**
    * Stack a list of tensors, the use must ensure that
    * the shapes of the tensors are appropriate for a stack
    * operation.
    *
    * @param inputs A sequence of tensors.
    *
    * @param axis The axis along which they should be stacked.
    *
    * @return The larger stacked tensor.
    * */
  def stack[D <: DataType](inputs: Seq[Tensor[D]], axis: Int = 0): Tensor[D] = tfi.stack(inputs, axis)

  /**
    * Split a tensor into a list of tensors.
    * */
  def unstack[D <: DataType](input: Tensor[D], number: Int = -1, axis: Int = 0): Seq[Tensor[D]] =
    tfi.unstack(input, number, axis)

  def concatenate[D <: DataType](inputs: Seq[Tensor[D]], axis: Tensor[INT32] = 0): Tensor[D] =
    tfi.concatenate(inputs, axis)

  /**
    * Generate a random tensor with independent and
    * identically distributed elements drawn from a
    * [[RandomVariable]] instance.
    * */
  def random[T, D <: DataType](
    dtype: D, shape: Int*)(
    rv: RandomVariable[T])(
    implicit ev: TensorConvertible.Aux[T, D]): Tensor[D] = {
    val buffer = rv.iid(shape.product).draw
    Tensor(dtype, buffer.head, buffer.tail:_*).reshape(Shape(shape:_*))
  }

  /**
    * Fill a tensor with a fixed value.
    * */
  def fill[T, D <: DataType](dataType: D, shape: Int*)(value: T)(implicit ev: SupportedType[T]): Tensor[D] =
    Tensor.fill(dataType, Shape(shape:_*))(value)

  def fill[T, D <: DataType](dataType: D, shape: Shape)(value: T)(implicit ev: SupportedType[T]): Tensor[D] =
    Tensor.fill(dataType, shape)(value)
}
