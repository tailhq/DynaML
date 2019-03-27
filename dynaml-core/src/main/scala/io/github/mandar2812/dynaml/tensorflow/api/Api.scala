package io.github.mandar2812.dynaml.tensorflow.api

import java.nio.ByteBuffer

import io.github.mandar2812.dynaml.probability._
import io.github.mandar2812.dynaml.pipes._
import org.platanios.tensorflow.api._


private[tensorflow] object Api {

  /**
    * Construct a tensor from a list of elements.
    *
    * @tparam D The type of the elements
    *
    * @param shape The shape of the tensor i.e. Shape(1,2,3)
    *              denotes a rank 3 tensor with 1, 2 and 3 dimensions
    *              for the ranks respectively.
    *
    * @param buffer The elements of type [[D]], can accept varying
    *               number of arguments.
    *
    * @return A tensorflow [[Tensor]] of the appropriate data type
    *         and shape.
    *
    * <b>Usage</b> dtf.tensor_from[Int](Shape(1, 2, 3))(1, 2, 3, 4, 5, 6)
    *
    * */
  def tensor_from[D: TF](shape: Shape)(buffer: D*): Tensor[D] = {
    val t: Tensor[D] = Tensor(buffer)
    t.reshape(shape)
  }

  /**
    * Construct a tensor from a list of elements.
    *
    * @tparam D The type of the elements
    *
    * @param shape The shape of the tensor given as any number
    *              of integer arguments.
    *
    * @param buffer The elements of type [[D]], as a Sequence
    *
    * @return A tensorflow [[Tensor]] of the appropriate data type
    *         and shape.
    *
    * <b>Usage</b> dtf.tensor_from[Int](1, 2, 3)((1 to 6).toSeq)
    *
    * */
  def tensor_from[D: TF](shape: Int*)(buffer: Seq[D]): Tensor[D] = {
    val t: Tensor[D] = Tensor[D](buffer)
    t.reshape(Shape(shape:_*))
  }

  /**
    * Construct a tensor from a array of bytes.
    *
    * @tparam D The type of the elements
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
  def tensor_from_buffer[D: TF](shape: Shape)(buffer: Array[Byte]): Tensor[D] = {
    Tensor.fromBuffer(shape, buffer.length.toLong, ByteBuffer.wrap(buffer))
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
  def tensor_from_buffer[D: TF](shape: Int*)(buffer: Array[Byte]): Tensor[D] =
    Tensor.fromBuffer[D](Shape(shape:_*), buffer.length.toLong, ByteBuffer.wrap(buffer))


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
  def tensor_i16(shape: Int*)(buffer: Short*): Tensor[Short] = 
    Tensor[Short](buffer).reshape(Shape(shape:_*))

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
  def tensor_i32(shape: Int*)(buffer: Int*): Tensor[Int] =
    Tensor[Int](buffer).reshape(Shape(shape:_*))

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
  def tensor_i64(shape: Int*)(buffer: Long*): Tensor[Long] = {
    val t: Tensor[Long] = Tensor[Long](buffer)
    t.reshape(Shape(shape:_*).toTensor)
  }

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
  def tensor_f16(shape: Int*)(buffer: Float*): Tensor[core.types.Half] = {

    val t: Tensor[Float] = Tensor[Float](buffer)

    t.reshape(Shape(shape:_*)).castTo[core.types.Half]
  }


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
  def tensor_f32(shape: Int*)(buffer: Float*): Tensor[Float] =
    Tensor[Float](buffer).reshape(Shape(shape:_*))

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
  def tensor_f64(shape: Int*)(buffer: Double*): Tensor[Double] =
    Tensor[Double](buffer).reshape(Shape(shape:_*))


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
  def stack[D: TF](inputs: Seq[Tensor[D]], axis: Int = 0): Tensor[D] = tfi.stack(inputs, axis)

  /**
    * Split a tensor into a list of tensors.
    * */
  def unstack[D: TF](input: Tensor[D], number: Int = -1, axis: Int = 0): Seq[Tensor[D]] =
    tfi.unstack(input, number, axis)

  def concatenate[D: TF](inputs: Seq[Tensor[D]], axis: Tensor[Int] = 0): Tensor[D] =
    tfi.concatenate(inputs, axis)

  /**
    * Generate a random tensor with independent and
    * identically distributed elements drawn from a
    * [[RandomVariable]] instance.
    * */
  def random[D: TF](shape: Int*)(
    rv: RandomVariable[D]): Tensor[D] = {
    val buffer = rv.iid(shape.product).draw
    Tensor[D](buffer).reshape(Shape(shape:_*))
  }

  /**
    * Fill a tensor with a fixed value.
    * */
  def fill[D: TF](shape: Int*)(value: D): Tensor[D] =
    Tensor.fill(Shape(shape:_*))(value)

  def fill[D: TF](shape: Shape)(value: D): Tensor[D] =
    Tensor.fill(shape)(value)

  /* def eig[D: TF: IsFloatOrDouble](matrices: Tensor[D]): Tensor[D] = {
    require(
      matrices.rank == 3 || matrices.rank == 2, 
      "In an eigen decomposition, the inputs must be [?, n, n] or [n, n]")

    val as = if(matrices.rank == 2) Seq(matrices) else matrices.unstack(axis = 0)

    require(
      as.head.shape(0) == as.head.shape(1), 
      "Only square matrices when using eig()")

    val s = as.head.shape(0).scalar
    
    def arnoldi(m: Tensor[D], n: Int): Tensor[D] = {
      val eps = Tensor(1E-12).castTo[D]

      val q0 = random[D](s)(GaussianRV(0.0, 1.0) > DataPipe[Double, D](_.asInstanceOf[D])).l2Normalize(axes = 0)

      

      m
    }



    matrices
  } */
}
