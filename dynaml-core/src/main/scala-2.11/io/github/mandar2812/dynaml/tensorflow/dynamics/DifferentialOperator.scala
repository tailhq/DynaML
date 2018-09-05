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
package io.github.mandar2812.dynaml.tensorflow.dynamics

import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.tensorflow.Learn
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.layers.Layer

/**
  * <h3>Differential Operator</h3>
  *
  * An abstract idea of a differential operator, it is
  * expressed as a [[DataPipe]] which takes a function/tensorflow layer
  * and returns another function/tensorflow layer.
  *
  * Differential operators are applied on some space of functions,
  * in DynaML (and TensorFlow), differentiable functions are
  * represented as computational layers.
  *
  * @tparam I Input domain of the underlying function space.
  * @tparam J Output domain of the underlying function space.
  * @param name A string identifier for the operator.
  * @author mandar2812
  *
  * */
sealed trait DifferentialOperator[I, J] extends DataPipe[Layer[I, J], Layer[I, J]] {

  val name: String

  def sources: Map[String, Option[DifferentialOperator[I, J]]]

  def +(other: DifferentialOperator[I, J]): DifferentialOperator[I, J]

  def -(other: DifferentialOperator[I, J]): DifferentialOperator[I, J]

  def *(const: J): DifferentialOperator[I, J]

  def *(layer: Layer[I, J]): DifferentialOperator[I, J]

  def *(other: DifferentialOperator[I, J]): DifferentialOperator[I, J]

  def apply(op: DifferentialOperator[I, J]): DifferentialOperator[I, J]

}

/**
  * A differential operator which operates on tensor valued functions.
  *
  * @tparam I Input domain of the underlying function space
  * @param name String identifier for the operator
  * @author mandar2812
  * */
abstract class TensorOperator[I](override val name: String) extends
  DifferentialOperator[I, Output] {

  self =>

  override def +(other: DifferentialOperator[I, Output]): DifferentialOperator[I, Output] =
    AddTensorOperator(self, other)


  override def -(other: DifferentialOperator[I, Output]): DifferentialOperator[I, Output] =
    AddTensorOperator(self, MultTensorOperator[I](Constant[I]("-1", -1), other))

  def *(const: Output): DifferentialOperator[I, Output] =
    MultTensorOperator(Constant[I](const.name, const), self)

  override def *(layer: Layer[I, Output]): DifferentialOperator[I, Output] =
    MultTensorOperator(self, SourceOperator(layer.name, layer))

  override def *(other: DifferentialOperator[I, Output]): DifferentialOperator[I, Output] =
    MultTensorOperator(self, other)

  override def apply(other: DifferentialOperator[I, Output]): TensorOperator[I] =
    ComposedOperator(self, other)
}


/**
  * A <i>source</i> or <i>injection</i> term is generally present
  * in the right hand side of PDE systems.
  *
  * T[f(x)] = q(x)
  * */
private[dynamics] case class SourceOperator[I, J](
  override val name: String,
  source: Layer[I, Output]) extends
  TensorOperator[I](name) {

  self =>

  override def run(data: Layer[I, Output]): Layer[I, Output] = source

  override def sources: Map[String, Option[DifferentialOperator[I, Output]]] = Map(self.name -> Some(self))
}

private[dynamics] case class Constant[I](override val name: String, t: Output) extends TensorOperator[I](name) {

  self =>

  override def run(data: Layer[I, Output]): Layer[I, Output] = Learn.constant[I](name, t)

  override def sources: Map[String, Option[DifferentialOperator[I, Output]]] = Map(self.name -> None)
}

/**
  * Composition of two operators.
  * */
private[dynamics] case class ComposedOperator[I](
  operator1: DifferentialOperator[I, Output],
  operator2: DifferentialOperator[I, Output]) extends
  TensorOperator[I](s"${operator1.name}[${operator2.name}]") {

  override def run(data: Layer[I, Output]): Layer[I, Output] = operator1.run(operator2.run(data))

  override def sources: Map[String, Option[DifferentialOperator[I, Output]]] = operator1.sources ++ operator2.sources
}

/**
  * A function (computational layer) multiplied to an operator.
  *
  * g(x) &times; T[.]
  *
  * @tparam I Input domain of the underlying function space
  * */
private[dynamics] case class MultTensorOperator[I](
  operator1: DifferentialOperator[I, Output],
  operator2: DifferentialOperator[I, Output]) extends
  TensorOperator[I](s"Mult[${operator1.name}, ${operator2.name}]") {

  override def run(data: Layer[I, Output]): Layer[I, Output] = {
    val layer1 = operator1(data)
    val layer2 = operator2(data)
    Learn.combined_layer(s"Combine[${layer1.name}, ${layer2.name}]", Seq(layer1, layer2)) >>
      Learn.mult_seq(s"Multiply[${layer1.name}, ${layer2.name}]")
  }

  override def sources: Map[String, Option[DifferentialOperator[I, Output]]] = operator1.sources ++ operator2.sources
}

/**
  * An operator which is the sum of two operators.
  *
  * T[.] = U[.] + V[.]
  *
  * @tparam I Input domain of the underlying function space
  * */
private[dynamics] case class AddTensorOperator[I](
  operator1: DifferentialOperator[I, Output],
  operator2: DifferentialOperator[I, Output]) extends
  TensorOperator[I](s"OperatorAdd[${operator1.name}, ${operator2.name}]") {

  override def run(data: Layer[I, Output]): Layer[I, Output] = {

    val layer1 = operator1(data)

    val layer2 = operator2(data)

    Learn.combined_layer(s"Combine[${layer1.name}, ${layer2.name}]", Seq(layer1, layer2)) >>
      Learn.sum_seq(s"Sum[${layer1.name}, ${layer2.name}]")
  }

  override def sources: Map[String, Option[DifferentialOperator[I, Output]]] = operator1.sources ++ operator2.sources

}