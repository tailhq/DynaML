/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
* */
package io.github.mandar2812.dynaml

import java.nio.ByteBuffer

import io.github.mandar2812.dynaml.probability._
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.tensors.{Tensor, TensorConvertible}
import org.platanios.tensorflow.api.types.{DataType, SupportedType}

/**
  * <h3>DynaML Tensorflow Utilities</h3>
  *
  * A collection of functions, transformations and
  * miscellaneous objects to help working with tensorflow
  * primitives and models.
  *
  * @author mandar2812 date: 23/11/2017
  * */
package object tensorflow {

  /**
    * <h4>DynaML Tensorflow Pointer</h4>
    * The [[dtf]] object is the entry point
    * for tensor related operations.
    * */
  object dtf {

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
    def tensor_from[T](dtype: DataType.Aux[T], shape: Shape)(buffer: T*)(implicit ev: TensorConvertible[T]): Tensor = {
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
    def tensor_from[T](dtype: String, shape: Int*)(buffer: Seq[T])(implicit ev: TensorConvertible[T]): Tensor = {
      Tensor(DataType.fromName(dtype), buffer.head, buffer.tail:_*).reshape(Shape(shape:_*))
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
    def tensor_from_buffer[T](
      dtype: DataType.Aux[T], shape: Shape)(
      buffer: Array[Byte]): Tensor = {
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
    def tensor_from_buffer(
      dtype: String, shape: Int*)(
      buffer: Array[Byte]): Tensor =
      Tensor.fromBuffer(
        DataType.fromName(dtype), Shape(shape:_*),
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
    def tensor_i16(shape: Int*)(buffer: Int*)(implicit ev: TensorConvertible[Double]) =
      Tensor(INT16, buffer.head, buffer.tail:_*).reshape(shape)

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
    def tensor_i32(shape: Int*)(buffer: Int*)(implicit ev: TensorConvertible[Double]) =
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
    def tensor_i64(shape: Int*)(buffer: Int*)(implicit ev: TensorConvertible[Double]) =
      Tensor(INT64, buffer.head, buffer.tail:_*).reshape(shape)

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
    def tensor_f16(shape: Int*)(buffer: Double*)(implicit ev: TensorConvertible[Double]) =
      Tensor(FLOAT16, buffer.head, buffer.tail:_*).reshape(shape)

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
    def tensor_f32(shape: Int*)(buffer: Double*)(implicit ev: TensorConvertible[Double]) =
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
    def tensor_f64(shape: Int*)(buffer: Double*)(implicit ev: TensorConvertible[Double]) =
      Tensor(FLOAT64, buffer.head, buffer.tail:_*).reshape(shape)

    /*def tensor_c64(shape: Int*)(buffer: Double*)(implicit ev: TensorConvertible[Double]) =
      Tensor(COMPLEX64, buffer.head, buffer.tail:_*).reshape(shape)

    def tensor_c128(shape: Int*)(buffer: Double*)(implicit ev: TensorConvertible[Double]) =
      Tensor(COMPLEX128, buffer.head, buffer.tail:_*).reshape(shape)*/

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
    def stack(inputs: Seq[Tensor], axis: Int = 0) = tfi.stack(inputs, axis)

    /**
      * Split a tensor into a list of tensors.
      * */
    def unstack(input: Tensor, number: Int = -1, axis: Int = 0) = tfi.unstack(input, number, axis)

    def concatenate(inputs: Seq[Tensor], axis: Tensor = 0): Tensor = tfi.concatenate(inputs, axis)

    /**
      * Generate a random tensor with independent and
      * identically distributed elements drawn from a
      * [[RandomVariable]] instance.
      * */
    def random[T](dtype: DataType.Aux[T], shape: Int*)(rv: RandomVariable[T])(implicit ev: TensorConvertible[T])
    : Tensor = {
      val buffer = rv.iid(shape.product).draw
      Tensor(dtype, buffer.head, buffer.tail:_*).reshape(Shape(shape:_*))
    }

    /**
      * Fill a tensor with a fixed value.
      * */
    def fill[T](dataType: DataType, shape: Int*)(value: T)(implicit ev: SupportedType[T]): Tensor =
      Tensor.fill(dataType, Shape(shape:_*))(value)

  }



}
