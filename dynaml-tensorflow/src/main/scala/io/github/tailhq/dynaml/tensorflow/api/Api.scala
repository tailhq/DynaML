package io.github.tailhq.dynaml.tensorflow.api

import java.nio.{ByteBuffer, ByteOrder}

import io.github.tailhq.dynaml.probability._
import io.github.tailhq.dynaml.pipes._
import org.platanios.tensorflow.api._

private[tensorflow] object Api {

  val tensorflow_version: String = org.platanios.tensorflow.jni.TensorFlow.version

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
    t.reshape(Shape(shape: _*))
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
    * <b>Usage</b> dtf.tensor_from_buffer[Float](Shape(1, 1), (1 to 4).toArray.map(_.toByte))
    *
    * */
  def tensor_from_buffer[D: TF](
    shape: Shape,
    buffer: Array[Byte]
  ): Tensor[D] = {
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
    * <b>Usage</b> dtf.tensor_from_buffer[Float](1, 1)((1 to 4).toArray.map(_.toByte))
    *
    * */
  def tensor_from_buffer[D: TF](shape: Int*)(buffer: Array[Byte]): Tensor[D] =
    Tensor.fromBuffer[D](
      Shape(shape: _*),
      buffer.length.toLong,
      ByteBuffer.wrap(buffer)
    )

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
    Tensor[Short](buffer).reshape(Shape(shape: _*))

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
    Tensor[Int](buffer).reshape(Shape(shape: _*))

  def buffer_i32(shape: Shape, buffer: Array[Int]): Tensor[Int] = {
    val byte_buff_size = 4 * buffer.length
    val byte_buffer =
      ByteBuffer.allocate(byte_buff_size).order(ByteOrder.nativeOrder())
    buffer.foreach(byte_buffer putInt _)

    tensor_from_buffer(shape, byte_buffer.array)
  }

  def sym_tensor_i32(shape: Int*)(buffer: Int*): Output[Int] =
    Output[Int](buffer).reshape(Shape(shape: _*))

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
    t.reshape(Shape(shape: _*).toTensor)
  }

  def buffer_i64(shape: Shape, buffer: Array[Long]): Tensor[Long] = {
    val byte_buff_size = 8 * buffer.length
    val byte_buffer =
      ByteBuffer.allocate(byte_buff_size).order(ByteOrder.nativeOrder())
    buffer.foreach(byte_buffer putLong _)

    tensor_from_buffer(shape, byte_buffer.array)
  }

  def sym_tensor_i64(shape: Int*)(buffer: Long*): Output[Long] =
    Output[Long](buffer).reshape(Shape(shape: _*))

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

    t.reshape(Shape(shape: _*)).castTo[core.types.Half]
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
    Tensor[Float](buffer).reshape(Shape(shape: _*))

  def buffer_f32(shape: Shape, buffer: Array[Float]): Tensor[Float] = {
    val byte_buff_size = 4 * buffer.length
    val byte_buffer =
      ByteBuffer.allocate(byte_buff_size).order(ByteOrder.nativeOrder())
    buffer.foreach(byte_buffer putFloat _)

    tensor_from_buffer(shape, byte_buffer.array)
  }

  def sym_tensor_f32(shape: Int*)(buffer: Float*): Output[Float] =
    Output[Float](buffer).reshape(Shape(shape: _*))

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
    Tensor[Double](buffer).reshape(Shape(shape: _*))

  def buffer_f64(shape: Shape, buffer: Array[Double]): Tensor[Double] = {
    val byte_buff_size = 8 * buffer.length
    val byte_buffer =
      ByteBuffer.allocate(byte_buff_size).order(ByteOrder.nativeOrder())
    buffer.foreach(byte_buffer putDouble _)

    tensor_from_buffer(shape, byte_buffer.array)
  }

  def sym_tensor_f64(shape: Int*)(buffer: Double*): Output[Double] =
    Output[Double](buffer).reshape(Shape(shape: _*))

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
  def stack[D: TF](inputs: Seq[Tensor[D]], axis: Int = 0): Tensor[D] =
    tfi.stack(inputs, axis)

  /**
    * Split a tensor into a list of tensors.
    * */
  def unstack[D: TF](
    input: Tensor[D],
    number: Int = -1,
    axis: Int = 0
  ): Seq[Tensor[D]] =
    tfi.unstack(input, number, axis)

  def concatenate[D: TF](
    inputs: Seq[Tensor[D]],
    axis: Tensor[Int] = 0
  ): Tensor[D] =
    tfi.concatenate(inputs, axis)

  /**
    * Generate a random tensor with independent and
    * identically distributed elements drawn from a
    * [[RandomVariable]] instance.
    * */
  def random[D: TF](shape: Int*)(rv: RandomVariable[D]): Tensor[D] = {
    val buffer = rv.iid(shape.product).draw.toSeq
    Tensor[D](buffer).reshape(Shape(shape: _*))
  }

  /**
    * Fill a tensor with a fixed value.
    * */
  def fill[D: TF](shape: Int*)(value: D): Tensor[D] =
    Tensor.fill(Shape(shape: _*))(value)

  def fill[D: TF](shape: Shape)(value: D): Tensor[D] =
    Tensor.fill(shape)(value)

  /**
    * This TC helps map between types and byte arrays.
    * It also keeps track of byte array sizes for various types.
    * todo: implement ByteSize typeclass for all tf types.
    */
  trait ByteSize[T] {
    def byteSize: Int
    def toBytes(in: T): Array[Byte]
  }
  object ByteSize {
    implicit val FloatByteSize: ByteSize[Float] = new ByteSize[Float] {
      override def byteSize: Int = 4

      override def toBytes(in: Float): Array[Byte] =
        java.nio.ByteBuffer.allocate(4).putFloat(in).array()
    }

    implicit val DoubleByteSize: ByteSize[Double] = new ByteSize[Double] {
      override def byteSize: Int = 8

      override def toBytes(in: Double): Array[Byte] =
        java.nio.ByteBuffer.allocate(8).putDouble(in).array()
    }

    implicit val IntByteSize: ByteSize[Int] = new ByteSize[Int] {
      override def byteSize: Int = 4

      override def toBytes(in: Int): Array[Byte] =
        java.nio.ByteBuffer.allocate(4).putInt(in).array()
    }

    implicit val LongByteSize: ByteSize[Long] = new ByteSize[Long] {
      override def byteSize: Int = 8

      override def toBytes(in: Long): Array[Byte] =
        java.nio.ByteBuffer.allocate(4).putLong(in).array()
    }
  }

  /**
    * This TC will help us determine whether a type is a nested array.
    * @tparam T the type of the nested array. eg., Array[Array[Float]]
    */
  trait NestedArray[T] {
    type DataType
  }
  object NestedArray extends NestedArrayLP {
    trait Aux[T, InnerData] extends NestedArray[T] {
      type DataType = InnerData
    }
    implicit def trivial[T] = new Aux[Array[T], T] {}
  }
  trait NestedArrayLP {
    implicit def complex[T, InnerData](
      implicit na: NestedArray.Aux[T, InnerData]
    ) =
      new NestedArray.Aux[Array[T], InnerData] {}
  }

  /**
    * The main tc for traversing a nested array, getting its tf byte size and shape.
    */
  trait BytesWithShape[T] {
    def bytesWithShape(in: T): (Array[Byte], Array[Int])
  }
  object BytesWithShape {
    implicit def bytesWithShape[T](implicit b: ByteSize[T]) = {
      new BytesWithShape[T] {
        override def bytesWithShape(in: T): (Array[Byte], Array[Int]) =
          (b.toBytes(in).reverse, Array()) //endianness
      }
    }

    implicit def nestedBytesWithShape[T](implicit bws: BytesWithShape[T]) =
      new BytesWithShape[Array[T]] {
        override def bytesWithShape(in: Array[T]): (Array[Byte], Array[Int]) = {
          val all = in.map(bws.bytesWithShape)
          //we require all internal shapes are identical to stack them.
          //todo: NAT support for compile time check.
          require(
            all.forall(_._2.sameElements(all.head._2)),
            "illegal array to tensor. peer arrays must be of same dim: " + all
              .map(_._2)
              .mkString(",")
          )
          all.flatMap(_._1) -> (Array(in.length) ++ all.head._2)
        }
      }
  }

  /**
    * Helper method to get tensors from nested array structures.
    */
  def toTensor[T, DataType](
    in: T
  )(
    implicit byteSize: ByteSize[DataType],
    na: NestedArray.Aux[T, DataType],
    bytesWithShape: BytesWithShape[T],
    ev: TF[DataType]
  ) = {
    val bws = bytesWithShape.bytesWithShape(in)
    Tensor.fromBuffer[DataType](
      Shape(bws._2),
      bws._2.product * byteSize.byteSize,
      ByteBuffer.wrap(bws._1)
    )
  }
}
