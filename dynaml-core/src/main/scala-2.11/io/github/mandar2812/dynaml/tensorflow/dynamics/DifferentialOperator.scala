package io.github.mandar2812.dynaml.tensorflow.dynamics

import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.tensorflow.Learn
import org.platanios.tensorflow.api.Output
import org.platanios.tensorflow.api.learn.layers.Layer

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