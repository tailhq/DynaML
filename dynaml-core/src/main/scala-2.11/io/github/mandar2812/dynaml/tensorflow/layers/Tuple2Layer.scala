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
package io.github.mandar2812.dynaml.tensorflow.layers

import org.platanios.tensorflow.api.core.types.TF
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api.{Output, tf}

/**
  * Combine two layers into a layer that accepts a tuple
  * containing each of their inputs.
  *
  * @tparam I1 Input type of the first layer
  * @tparam O1 Output type of the first layer
  * @tparam I2 Input type of the second layer
  * @tparam O2 Output type of the second layer
  *
  * @param name Name/String identifier of the layer
  *
  * @author mandar2812 date 31/05/2018
  * */
case class Tuple2Layer[I1, O1, I2, O2](override val name: String, layer1: Layer[I1, O1], layer2: Layer[I2, O2])
  extends Layer[(I1, I2), (O1, O2)](name) {

  override val layerType: String = s"TupleLayer[${layer1.layerType}, ${layer2.layerType}]"

  val _1: Layer[I1, O1] = layer1
  val _2: Layer[I2, O2] = layer2

  override def forwardWithoutContext(input: (I1, I2))(implicit mode: Mode): (O1, O2) =
    (layer1.forward(input._1)(mode), layer2.forward(input._2)(mode))
}

/**
  * Take a tuple of symbolic tensors and concatenate them along a
  * specified axis.
  *
  * @author mandar2812 date 31/05/2018
  * */
case class ConcatenateTuple2[T: TF](override val name: String, axis: Int)
  extends Layer[(Output[T], Output[T]), Output[T]](name) {

  override val layerType: String = s"ConcatTuple2"

  override def forwardWithoutContext(input: (Output[T], Output[T]))(implicit mode: Mode): Output[T] =
    tf.concatenate(Seq(input._1, input._2), axis)
}

/**
  * Take a tuple of symbolic tensors and stack them along a
  * specified axis.
  *
  * @author mandar2812 date 31/05/2018
  * */
case class StackTuple2[T: TF](override val name: String, axis: Int)
  extends Layer[(Output[T], Output[T]), Output[T]](name) {

  override val layerType: String = s"StackTuple2"

  override def forwardWithoutContext(input: (Output[T], Output[T]))(implicit mode: Mode): Output[T] =
    tf.stack(Seq(input._1, input._2), axis)
}
