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
abstract class DifferentialOperator[I, J](val name: String) extends DataPipe[Layer[I, J], Layer[I, J]] {

  def +(other: DifferentialOperator[I, J]): DifferentialOperator[I, J]

  def -(other: DifferentialOperator[I, J]): DifferentialOperator[I, J]

  def *(const: J): DifferentialOperator[I, J]

  def *(layer: Layer[I, J]): DifferentialOperator[I, Output]

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
  DifferentialOperator[I, Output](name) {

  self =>

  override def +(other: DifferentialOperator[I, Output]): DifferentialOperator[I, Output] =
    AddTensorOperator(self, other)


  override def -(other: DifferentialOperator[I, Output]): DifferentialOperator[I, Output] = AddTensorOperator(
    self,
    ConstMultTensorOperator(
      -1.0,
      other)
  )

  def *(const: Output): DifferentialOperator[I, Output] = ConstMultTensorOperator(const, self)

  override def *(layer: Layer[I, Output]): DifferentialOperator[I, Output] = MultTensorOperator(layer, self)

  override def apply(other: DifferentialOperator[I, Output]): TensorOperator[I] =
    ComposedOperator(self, other)
}

/**
  * Composition of two operators.
  * */
case class ComposedOperator[I](
  operator1: DifferentialOperator[I, Output],
  operator2: DifferentialOperator[I, Output]) extends
  TensorOperator[I](s"${operator1.name}[${operator2.name}]") {

  override def run(data: Layer[I, Output]): Layer[I, Output] = operator1.run(operator2.run(data))
}

/**
  * A constant (tensor) multiplied to an operator.
  *
  * &alpha; &times; T[.]
  *
  * @tparam I Input domain of the underlying function space
  * */
case class ConstMultTensorOperator[I](const: Output, operator: DifferentialOperator[I, Output]) extends
  TensorOperator[I](s"ScalarMult[${const.toString()}, ${operator.name}]") {

  override def run(data: Layer[I, Output]): Layer[I, Output] = {
    val layer = operator(data)
    layer >> Learn.multiply_const(const, s"${layer.name}")
  }
}

/**
  * A function (computational layer) multiplied to an operator.
  *
  * g(x) &times; T[.]
  *
  * @tparam I Input domain of the underlying function space
  * */
case class MultTensorOperator[I](
  function: Layer[I, Output],
  operator: DifferentialOperator[I, Output]) extends
  TensorOperator[I](s"Mult[${function.name}, ${operator.name}]") {

  override def run(data: Layer[I, Output]): Layer[I, Output] = {
    val layer = operator(data)
    Learn.combined_layer(s"Combine[${function.name}, ${layer.name}]", Seq(function, layer)) >>
      Learn.mult_seq(s"Multiply[${function.name}, ${layer.name}]")
  }
}

/**
  * An operator which is the sum of two operators.
  *
  * T[.] = U[.] + V[.]
  *
  * @tparam I Input domain of the underlying function space
  * */
case class AddTensorOperator[I](
  operator1: DifferentialOperator[I, Output],
  operator2: DifferentialOperator[I, Output]) extends
  TensorOperator[I](s"OperatorAdd[${operator1.name}, ${operator2.name}]") {

  override def run(data: Layer[I, Output]): Layer[I, Output] = {

    val layer1 = operator1(data)

    val layer2 = operator2(data)

    Learn.combined_layer(s"Combine[${layer1.name}, ${layer2.name}]", Seq(layer1, layer2)) >>
      Learn.sum_seq(s"Sum[${layer1.name}, ${layer2.name}]")
  }

}

