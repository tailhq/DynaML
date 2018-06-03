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

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer

/**
  * Stacks output produced by a tensorflow concatenation layer
  *
  * @param axis The axis over which the tensors should be concatenated,
  *             defaults to -1
  *
  * @author mandar2812 date 2018/03/16
  * */
case class StackOutputs(override val name: String, axis: Int = -1) extends Layer[Seq[Output], Output](name) {

  override val layerType: String = s"Stack[axis:$axis]"

  override protected def _forward(input: Seq[Output], mode: Mode): Output = tf.stack(input, axis)

}


case class Unstack(override val name: String, axis: Int = -1) extends Layer[Output, Seq[Output]](name) {

  override val layerType: String = s"Unstack[axis:$axis]"

  override protected def _forward(input: Output, mode: Mode): Seq[Output] = tf.unstack(input, axis)
}


/**
  * Stacks output produced by a tensorflow concatenation layer
  *
  * @param axis The axis over which the tensors should be concatenated,
  *             defaults to -1
  *
  * @author mandar2812 date 2018/06/03
  * */
case class ConcatenateOutputs(override val name: String, axis: Int = -1) extends Layer[Seq[Output], Output](name) {

  override val layerType: String = s"Concatenate[axis:$axis]"

  override protected def _forward(input: Seq[Output], mode: Mode): Output = tf.concatenate(input, axis)

}

/**
  * Combine a collection of layers into a layer which accepts
  * sequences of symbolic tensors.
  *
  * @param layers The layers to be joined.
  *
  * @author mandar2812 date 2018/03/16
  * */
case class StackLayers[T, R](override val name: String, layers: Seq[Layer[T, R]]) extends Layer[Seq[T], Seq[R]](name) {
  override val layerType: String = s"LayerStack[${layers.map(_.name).mkString(",")}]"

  override protected def _forward(input: Seq[T], mode: Mode): Seq[R] =
    layers.zip(input).map(c => c._1.forward(c._2, mode))
}


case class IdentityLayer[I](override val name: String) extends Layer[I, I](name) {

  override val layerType: String = s"Identity"

  override protected def _forward(input: I, mode: Mode): I = input

}


