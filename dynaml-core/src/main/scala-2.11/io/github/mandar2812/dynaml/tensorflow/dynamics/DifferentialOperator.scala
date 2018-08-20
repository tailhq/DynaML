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
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer

/**
  * An abstract idea of a differential operator, it is
  * expressed as a [[DataPipe]] which takes a function/tensorflow layer
  * and returns another function/tensorflow layer.
  *
  * */
abstract class DifferentialOperator[I, J](val name: String) extends DataPipe[Layer[I, J], Layer[I, J]] {

  def +(other: DifferentialOperator[I, J]): DifferentialOperator[I, J]

  def -(other: DifferentialOperator[I, J]): DifferentialOperator[I, J]

  def *(const: I): DifferentialOperator[I, J]

  def *(layer: Layer[Output, Output]): DifferentialOperator[Output, Output]

  def apply(op: DifferentialOperator[I, J]): DifferentialOperator[I, J]

}

abstract class TensorOperator(override val name: String) extends DifferentialOperator[Output, Output](name) {

  self =>

  override def +(other: DifferentialOperator[Output, Output]): TensorOperator =
    AddTensorOperator(self, other)


  override def -(other: DifferentialOperator[Output, Output]): TensorOperator = AddTensorOperator(
    self,
    ConstMultTensorOperator(
      -1.0,
      other)
  )

  override def *(const: Output): DifferentialOperator[Output, Output] = ConstMultTensorOperator(const, self)

  override def *(layer: Layer[Output, Output]): DifferentialOperator[Output, Output] = MultTensorOperator(layer, self)

  override def apply(other: DifferentialOperator[Output, Output]): DifferentialOperator[Output, Output] =
    ComposedOperator(self, other)
}

case class ComposedOperator(
  operator1: DifferentialOperator[Output, Output],
  operator2: DifferentialOperator[Output, Output]) extends
  TensorOperator(s"${operator1.name}[${operator2.name}]") {

  override def run(data: Layer[Output, Output]): Layer[Output, Output] = operator1.run(operator2.run(data))
}

case class ConstMultTensorOperator(const: Output, operator: DifferentialOperator[Output, Output]) extends
  TensorOperator(s"ScalarMult[${const.toString()}, ${operator.name}]") {

  override def run(data: Layer[Output, Output]): Layer[Output, Output] = {
    val layer = operator(data)
    layer >> Learn.multiply_const(const, s"${layer.name}")
  }
}

case class MultTensorOperator(
  function: Layer[Output, Output],
  operator: DifferentialOperator[Output, Output]) extends
  TensorOperator(s"Mult[${function.name}, ${operator.name}]") {

  override def run(data: Layer[Output, Output]): Layer[Output, Output] = {
    val layer = operator(data)
    Learn.combined_layer(s"Combine[${function.name}, ${layer.name}]", Seq(function, layer)) >>
      Learn.mult_seq(s"Multiply[${function.name}, ${layer.name}]")
  }
}

case class AddTensorOperator(
  operator1: DifferentialOperator[Output, Output], operator2: DifferentialOperator[Output, Output]) extends
  TensorOperator(s"OperatorAdd[${operator1.name}, ${operator2.name}]") {

  override def run(data: Layer[Output, Output]): Layer[Output, Output] = {

    val layer1 = operator1(data)

    val layer2 = operator2(data)

    Learn.combined_layer(s"Combine[${layer1.name}, ${layer2.name}]", Seq(layer1, layer2)) >>
      Learn.sum_seq(s"Sum[${layer1.name}, ${layer2.name}]")
  }

}

private[dynamics] object Gradient extends TensorOperator(s"Grad") {

  override def run(data: Layer[Output, Output]): Layer[Output, Output] = {

    new Layer[Output, Output](s"Grad[${data.name}]") {

      override val layerType: String = "GradLayer"

      override protected def _forward(input: Output)(implicit mode: Mode): Output = {

        val output = data.forward(input)


        def gradTRec(y: Seq[Output], x: Output, rank: Int): Output = rank match {
          case 1 =>
            tf.stack(
              y.map(
                o => tf.gradients.gradients(Seq(o), Seq(x)).head.toOutput
              ), -1)
          case _ =>
            gradTRec(y.flatMap(_.unstack(-1, -1)), x, rank - 1)

        }

        gradTRec(Seq(output), input, output.rank).reshape(output.shape ++ input.shape(1::))

      }
    }
  }

}
