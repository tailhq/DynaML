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

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.tensors.{Context, Tensor, TensorConvertible}
import org.platanios.tensorflow.api.types.DataType

import scala.util.DynamicVariable


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

  object dtf {

    def tensor_from[T](dtype: DataType.Aux[T], shape: Shape)(buffer: T*)(implicit ev: TensorConvertible[T]): Tensor = {
      Tensor(dtype, buffer.head, buffer.tail:_*).reshape(shape)
    }

    def tensor_from[T](dtype: String, shape: Int*)(buffer: Seq[T])(implicit ev: TensorConvertible[T]): Tensor = {
      Tensor(DataType.fromName(dtype), buffer.head, buffer.tail:_*).reshape(Shape(shape:_*))
    }

    def tensor_i16(shape: Int*)(buffer: Double*)(implicit ev: TensorConvertible[Double]) =
      Tensor(INT16, buffer.head, buffer.tail:_*).reshape(shape)

    def tensor_i32(shape: Int*)(buffer: Double*)(implicit ev: TensorConvertible[Double]) =
      Tensor(INT32, buffer.head, buffer.tail:_*).reshape(shape)

    def tensor_i64(shape: Int*)(buffer: Double*)(implicit ev: TensorConvertible[Double]) =
      Tensor(INT64, buffer.head, buffer.tail:_*).reshape(shape)

    def tensor_f16(shape: Int*)(buffer: Double*)(implicit ev: TensorConvertible[Double]) =
      Tensor(FLOAT16, buffer.head, buffer.tail:_*).reshape(shape)

    def tensor_f32(shape: Int*)(buffer: Double*)(implicit ev: TensorConvertible[Double]) =
      Tensor(FLOAT32, buffer.head, buffer.tail:_*).reshape(shape)

    def tensor_f64(shape: Int*)(buffer: Double*)(implicit ev: TensorConvertible[Double]) =
      Tensor(FLOAT64, buffer.head, buffer.tail:_*).reshape(shape)

    def tensor_c64(shape: Int*)(buffer: Double*)(implicit ev: TensorConvertible[Double]) =
      Tensor(COMPLEX64, buffer.head, buffer.tail:_*).reshape(shape)

    def tensor_c128(shape: Int*)(buffer: Double*)(implicit ev: TensorConvertible[Double]) =
      Tensor(COMPLEX128, buffer.head, buffer.tail:_*).reshape(shape)

    def stack(inputs: Seq[Tensor], axis: Int = 0) = tfi.stack(inputs, axis)

    def unstack(input: Tensor, number: Int = -1, axis: Int = 0) = tfi.unstack(input, number, axis)

  }



}
