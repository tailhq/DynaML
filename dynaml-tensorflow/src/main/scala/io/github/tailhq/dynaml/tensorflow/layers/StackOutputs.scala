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
package io.github.tailhq.dynaml.tensorflow.layers

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, TF}
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer

import scala.reflect.ClassTag
import org.platanios.tensorflow.api.learn.layers.Activation

/**
  * Stacks output produced by a tensorflow concatenation layer
  *
  * @param axis The axis over which the tensors should be concatenated,
  *             defaults to -1
  *
  * @author tailhq date 2018/03/16
  * */
case class StackOutputs[D: TF](
  override val name: String,
  axis: Int = -1) extends
  Layer[Seq[Output[D]], Output[D]](name) {

  override val layerType: String = s"Stack[axis:$axis]"

  override def forwardWithoutContext(input: Seq[Output[D]])(implicit mode: Mode): Output[D] = tf.stack(input, axis)

}


case class Unstack[D: TF](
  override val name: String,
  axis: Int = -1) extends
  Layer[Output[D], Seq[Output[D]]](name) {

  override val layerType: String = s"Unstack[axis:$axis]"

  override def forwardWithoutContext(input: Output[D])(implicit mode: Mode): Seq[Output[D]] = tf.unstack(input, axis)
}


/**
  * Stacks output produced by a tensorflow concatenation layer
  *
  * @param axis The axis over which the tensors should be concatenated,
  *             defaults to -1
  *
  * @author tailhq date 2018/06/03
  * */
case class ConcatenateOutputs[D: TF](
  override val name: String,
  axis: Int = -1) extends
  Layer[Seq[Output[D]], Output[D]](name) {

  override val layerType: String = s"Concatenate[axis:$axis]"

  override def forwardWithoutContext(input: Seq[Output[D]])(implicit mode: Mode): Output[D] =
    tf.concatenate(input, axis)

}

case class SumSeq[D: TF: IsNotQuantized](
  override val name: String) extends
  Layer[Seq[Output[D]], Output[D]](name) {

  override val layerType: String = s"SumSeq"

  override def forwardWithoutContext(input: Seq[Output[D]])(implicit mode: Mode): Output[D] =
    input.reduceLeft(tf.add(_, _))
}

case class MultSeq[D: TF: IsNotQuantized](
  override val name: String) extends
  Layer[Seq[Output[D]], Output[D]](name) {

  override val layerType: String = s"MultSeq"

  override def forwardWithoutContext(input: Seq[Output[D]])(implicit mode: Mode): Output[D] = 
    input.reduceLeft(tf.multiply(_, _))
}


case class SumTuple[D: TF: IsNotQuantized](
  override val name: String)
  extends Layer[(Output[D], Output[D]), Output[D]](name) {

  override val layerType: String = s"Sum"

  override def forwardWithoutContext(input: (Output[D], Output[D]))(implicit mode: Mode): Output[D] = 
    tf.add(input._1, input._2)
}

/**
  * Combine a collection of layers into a layer which accepts
  * sequences of symbolic tensors.
  *
  * @param layers The layers to be joined.
  *
  * @author tailhq date 2018/03/16
  * */
case class SeqLayer[T, R](
  override val name: String,
  layers: Seq[Layer[T, R]]) extends
  Layer[Seq[T], Seq[R]](name) {
  override val layerType: String = s"SeqLayer[${layers.map(_.layerType).mkString(",")}]"

  override def forwardWithoutContext(input: Seq[T])(implicit mode: Mode): Seq[R] =
    layers.zip(input).map(c => c._1.forward(c._2)(mode))
}

case class ArrayLayer[T, R: ClassTag](
  override val name: String,
  layers: Array[Layer[T, R]]) extends
  Layer[Array[T], Array[R]](name) {
  override val layerType: String = s"ArrayLayer[${layers.map(_.layerType).mkString(",")}]"

  override def forwardWithoutContext(input: Array[T])(implicit mode: Mode): Array[R] =
    layers.zip(input).map(c => c._1.forward(c._2)(mode))
}


case class MapLayer[K, T, R: ClassTag](
  override val name: String,
  layers: Map[K, Layer[T, R]]) extends
  Layer[T, Map[K, R]](name) {

  /*require(
    layers.nonEmpty,
    "In a Map Layer, there must be atleast one key-value layer tuple.")*/

  override val layerType: String = s"Map[${layers.map(kv => (kv._1.toString, kv._2.layerType)).mkString(",")}]"

  override def forwardWithoutContext(input: T)(implicit mode: Mode): Map[K, R] =
    layers.map(c => (c._1, c._2.forwardWithoutContext(input)))
}

case class ScopedMapLayer[K, T, R: ClassTag](
  override val name: String,
  layers: Map[K, Layer[T, R]],
  scopes: Seq[String]) extends
  Layer[T, Map[K, R]](name) {

  require(
    scopes.length == layers.size,
    "Number of scopes in a Scoped Map Layer must be equal to the number of layers")

  override val layerType: String =
    s"ScopedMap[${layers.map(kv => (kv._1.toString, kv._2.layerType)).mkString(",")}]"

  override def forwardWithoutContext(input: T)(implicit mode: Mode): Map[K, R] =
    layers.zip(scopes).map(c =>
      tf.updatedVariableScope(
        variableScope = tf.currentVariableScope.copy(name = c._2),
        reuse = tf.ReuseExistingVariableOnly){
        (c._1._1, c._1._2.forwardWithoutContext(input))
      })
}


/**
  * Combine a collection of layers into a layer which maps
  * its input through each layer in the sequence.
  *
  * @param layers The layers to be joined.
  *
  * @author tailhq date 2018/03/16
  * */
case class CombinedLayer[T, R](
  override val name: String,
  layers: Seq[Layer[T, R]])
  extends Layer[T, Seq[R]](name) {
  override val layerType: String = s"CombinedLayer[${layers.map(_.layerType).mkString(",")}]"

  override def forwardWithoutContext(input: T)(implicit mode: Mode): Seq[R] =
    layers.map(_.forward(input)(mode))
}

case class CombinedArrayLayer[T, R: ClassTag](override val name: String, layers: Array[Layer[T, R]])
  extends Layer[T, Array[R]](name) {
  override val layerType: String = s"CombinedArrayLayer[${layers.map(_.layerType).mkString(",")}]"

  override def forwardWithoutContext(input: T)(implicit mode: Mode): Array[R] =
    layers.map(_.forward(input)(mode)).toArray
}


case class IdentityLayer[I](override val name: String) extends Layer[I, I](name) {

  override val layerType: String = s"Identity"

  override def forwardWithoutContext(input: I)(implicit mode: Mode): I = input

}

case class IdentityAct[I: TF](override val name: String) extends Activation[I](name) {

  override val layerType: String = s"Identity"

  override def forwardWithoutContext(input: Output[I])(implicit mode: Mode): Output[I] = input

}


case class MultConstant[D: TF: IsNotQuantized](const: Output[D], override val name: String)
  extends Layer[Output[D], Output[D]](name) {
  override val layerType: String = s"MultConst"

  override def forwardWithoutContext(input: Output[D])(implicit mode: Mode): Output[D] = tf.multiply(input, const)
}

