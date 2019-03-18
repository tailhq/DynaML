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

import io.github.mandar2812.dynaml.tensorflow.api.Api
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, TF}
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.variables.Initializer

/**
  * <h3>Differential Operators & PDEs</h3>
  *
  * <i>Partial Differential Equations</i> (PDE) are the most important tool for
  * modelling physical processes and phenomena which vary in space and time.
  *
  * The dynamics package contains classes and implementations of PDE operators.
  *
  * */
private[tensorflow] trait DynamicsAPI {

  type Operator[I, J]                               = DifferentialOperator[I, J]
  type TensorOp[I, T]                               = TensorOperator[I, T]

  val I: IdentityOperator.type                      = IdentityOperator
  val jacobian: Gradient[Float]                     = Gradient[Float]
  val ∇ : Gradient[Float]                           = Gradient[Float]
  val hessian: TensorOp[Output[Float], Float]       = ∇(∇)
  val source: SourceOperator.type                   = SourceOperator
  val constant: Constant.type                       = Constant


  def one[I, D: TF: IsNotQuantized](shape: Shape): Constant[I, D] =
    constant[I, D]("One", Tensor.ones[D](shape))

  def variable[I, D: TF: IsNotQuantized](
    name: String,
    shape: Shape,
    initializer: Initializer = null): SourceOperator[I, D] =
    source(
      name,
      new Layer[I, Output[D]](name) {
        override val layerType: String = "Quantity"

        override def forwardWithoutContext(input: I)(implicit mode: Mode): Output[D] = {
          val quantity = getParameter(name = "value", shape, initializer)
          quantity
        }
      })

  def divergence[D: TF: IsNotQuantized](
    dim: Int, inputSlices: Seq[Indexer] = Seq(---)): TensorOperator[Output[D], D] = {

    val Id = constant[Output[D], D](
      s"IdentityMat($dim)",
      Api.tensor_i32(dim, dim)(
        Seq.tabulate[Int](dim, dim)((i, j) => if(i == j) 1 else 0).flatten:_*
      ).castTo[D]
    )

    TensorDotOperator[Output[D], D](
      Id,
      SlicedGradient(
        name = s"Grad")(
        inputSlices:_*)(---),
      Seq(0, 1),
      Seq(1, 2))
  }

  def div[D: TF: IsNotQuantized]: TensorOperator[Output[D], D] = divergence(dim = 3, Seq(1::))


  /**
    * Calculate a `sliced` gradient.
    *
    * ∂f[slices]/∂x[slices]
    *
    * @param name A string identifier for the operator
    * @param input_slices Determines over which subset of the inputs to compute the gradient
    * @param output_slices Determines over which subset of the outputs to compute the gradient
    * */
  def ∂[D: TF: IsNotQuantized](
    name: String)(
    input_slices: Indexer*)(
    output_slices: Indexer*): SlicedGradient[D] = SlicedGradient(name)(input_slices:_*)(output_slices:_*)

  /**
    * Time derivative (∂f/∂t) of a function f(t, s),
    * which accepts space-time vectors [t, s_1, s_2, ..., s_n]
    * as inputs
    * */
  def d_t: SlicedGradient[Float]                = ∂("D_t")(0)(---)

  /**
    * Space derivative (∂f/∂s) of a function f(t, s),
    * which accepts space-time vectors [t, s_1, s_2, ..., s_n]
    * as inputs
    * */
  def d_s: SlicedGradient[Float]                = ∂("D_s")( 1 ::)(---)

}
