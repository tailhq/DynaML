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
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, TF}
import org.platanios.tensorflow.api.learn.Mode
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
  * @author mandar2812
  *
  * */
sealed trait DifferentialOperator[I, J] extends DataPipe[Layer[I, J], Layer[I, J]] {

  self =>

  /**
    * A string identifier for the operator.
    * */
  val name: String

  protected[dynamics] def sources: Map[String, Option[DifferentialOperator[I, J]]]

  lazy val variables: Map[String, DifferentialOperator[I, J]] = for {
    kv <- self.sources
    if kv._2.isDefined
    s <- kv._2
    key = kv._1
  } yield (key, s)

  def +(other: DifferentialOperator[I, J]): DifferentialOperator[I, J]

  def -(other: DifferentialOperator[I, J]): DifferentialOperator[I, J]

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
abstract class TensorOperator[I, D: TF: IsNotQuantized](override val name: String) extends
  DifferentialOperator[I, Output[D]] {

  self =>

  override def +(other: DifferentialOperator[I, Output[D]]): DifferentialOperator[I, Output[D]] =
    AddTensorOperator(self, other)


  override def -(other: DifferentialOperator[I, Output[D]]): DifferentialOperator[I, Output[D]] =
    AddTensorOperator(self, MultTensorOperator[I, D](Constant[I, D]("-1", Tensor(-1).castTo[D]), other))

  def *(const: Tensor[D]): DifferentialOperator[I, Output[D]] =
    MultTensorOperator(Constant[I, D](const.name, const), self)

  override def *(layer: Layer[I, Output[D]]): DifferentialOperator[I, Output[D]] =
    MultTensorOperator(self, SourceOperator(layer.name, layer))

  override def *(other: DifferentialOperator[I, Output[D]]): DifferentialOperator[I, Output[D]] =
    MultTensorOperator(self, other)

  def x(other: DifferentialOperator[I, Output[D]]): DifferentialOperator[I, Output[D]] =
    MatrixMultOperator(self, other)

  override def apply(other: DifferentialOperator[I, Output[D]]): TensorOperator[I, D] =
    ComposedOperator(self, other)
}


private[dynamics] case class IdentityOperator[D: TF: IsNotQuantized](
  override val name: String) extends
  TensorOperator[Output[D], D](name) {

  self =>

  override protected[dynamics] def sources: Map[String, Option[DifferentialOperator[Output[D], Output[D]]]] =
    Map(self.name -> None)

  override def run(data: Layer[Output[D], Output[D]]): Layer[Output[D], Output[D]] = data
}

/**
  * A <i>source</i> or <i>injection</i> term is generally present
  * in the right hand side of PDE systems.
  *
  * T[f(x)] = q(x)
  * */
private[dynamics] case class SourceOperator[I, D: TF: IsNotQuantized](
  override val name: String,
  source: Layer[I, Output[D]],
  isSystemVariable: Boolean = true) extends
  TensorOperator[I, D](name) {

  self =>

  override def run(data: Layer[I, Output[D]]): Layer[I, Output[D]] = source

  protected[dynamics] override def sources: Map[String, Option[DifferentialOperator[I, Output[D]]]] =
    if(isSystemVariable) Map(self.name -> Some(self))
    else Map(self.name -> None)
}

private[dynamics] case class Constant[I, D: TF: IsNotQuantized](
  override val name: String,
  t: Tensor[D]) extends
  TensorOperator[I, D](name) {

  self =>

  override def run(data: Layer[I, Output[D]]): Layer[I, Output[D]] = Learn.constant[I, D](name, t)

  protected[dynamics] override def sources: Map[String, Option[DifferentialOperator[I, Output[D]]]] =
    Map(self.name -> None)
}

/**
  * Composition of two operators.
  * */
private[dynamics] case class ComposedOperator[I, D: TF: IsNotQuantized](
  operator1: DifferentialOperator[I, Output[D]],
  operator2: DifferentialOperator[I, Output[D]]) extends
  TensorOperator[I, D](s"${operator1.name}[${operator2.name}]") {

  override def run(data: Layer[I, Output[D]]): Layer[I, Output[D]] = operator1.run(operator2.run(data))

  protected[dynamics] override def sources: Map[String, Option[DifferentialOperator[I, Output[D]]]] =
    operator1.sources ++ operator2.sources
}

/**
  * A function (computational layer) multiplied to an operator.
  *
  * g(x) &times; T[.]
  *
  * @tparam I Input domain of the underlying function space
  * */
private[dynamics] case class MultTensorOperator[I, D: TF: IsNotQuantized](
  operator1: DifferentialOperator[I, Output[D]],
  operator2: DifferentialOperator[I, Output[D]]) extends
  TensorOperator[I, D](s"Mult[${operator1.name}, ${operator2.name}]") {

  override def run(data: Layer[I, Output[D]]): Layer[I, Output[D]] = {
    val layer1 = operator1(data)
    val layer2 = operator2(data)
    Learn.combined_layer(s"OperatorMult_${layer1.name}-${layer2.name}_", Seq(layer1, layer2)) >>
      Learn.mult_seq(s"MultSeq_${layer1.name}-${layer2.name}_")
  }

  protected[dynamics] override def sources: Map[String, Option[DifferentialOperator[I, Output[D]]]] =
    operator1.sources ++ operator2.sources
}

/**
  * An operator which is the sum of two operators.
  *
  * T[.] = U[.] + V[.]
  *
  * @tparam I Input domain of the underlying function space
  * */
private[dynamics] case class AddTensorOperator[I, D: TF: IsNotQuantized](
  operator1: DifferentialOperator[I, Output[D]],
  operator2: DifferentialOperator[I, Output[D]]) extends
  TensorOperator[I, D](s"OperatorAdd[${operator1.name}, ${operator2.name}]") {

  override def run(data: Layer[I, Output[D]]): Layer[I, Output[D]] = {

    val layer1 = operator1(data)

    val layer2 = operator2(data)

    Learn.combined_layer(s"OperatorAdd_${layer1.name}-${layer2.name}_", Seq(layer1, layer2)) >>
      Learn.sum_seq(s"SumSeq_${layer1.name}-${layer2.name}_")
  }

  protected[dynamics] override def sources: Map[String, Option[DifferentialOperator[I, Output[D]]]] =
    operator1.sources ++ operator2.sources

}

private[dynamics] case class MatrixMultOperator[I, D: TF: IsNotQuantized](
  operator1: DifferentialOperator[I, Output[D]],
  operator2: DifferentialOperator[I, Output[D]])
  extends TensorOperator[I, D](s"MatMul[${operator1.name}, ${operator2.name}]") {

  override protected[dynamics] def sources: Map[String, Option[DifferentialOperator[I, Output[D]]]] =
    operator1.sources ++ operator2.sources

  override def run(data: Layer[I, Output[D]]): Layer[I, Output[D]] = {

    val layer1 = operator1(data)
    val layer2 = operator2(data)

    Learn.combined_layer(s"MatMul_${layer1.name}-${layer2.name}_", Seq(layer1, layer2)) >>
      new Layer[Seq[Output[D]], Output[D]](s"MatMulOp_${layer1.name}_${layer2.name}_") {

        override val layerType: String = "MatMul"

        override def forwardWithoutContext(input: Seq[Output[D]])(implicit mode: Mode): Output[D] =
          tf.matmul(input.head, input.last)
      }

  }
}

private[dynamics] case class TensorDotOperator[I, D: TF: IsNotQuantized](
  operator1: DifferentialOperator[I, Output[D]],
  operator2: DifferentialOperator[I, Output[D]],
  axes1: Seq[Int], axes2: Seq[Int]) extends
  TensorOperator[I, D](s"TensorDot[(${operator1.name}, $axes1)," + s"(${operator2.name}, $axes2)]") {

  override protected[dynamics] def sources: Map[String, Option[DifferentialOperator[I, Output[D]]]] =
    operator1.sources ++ operator2.sources

  override def run(data: Layer[I, Output[D]]): Layer[I, Output[D]] = {

    val (layer1, layer2) = (operator1(data), operator2(data))

    Learn.combined_layer(s"TensorDot_${layer1.name}-${layer2.name}_", Seq(layer1, layer2)) >>
      new Layer[Seq[Output[D]], Output[D]](s"TensorDotOp_${layer1.name}_${layer2.name}_") {
        override val layerType: String = "TensorDot"

        override def forwardWithoutContext(input: Seq[Output[D]])(implicit mode: Mode): Output[D] =
          tf.tensorDot(input.head, input.last, axes1, axes2)
      }

  }
}