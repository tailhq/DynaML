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
package io.github.tailhq.dynaml.kernels

import spire.algebra.Field
import breeze.linalg.DenseMatrix
import io.github.tailhq.dynaml.pipes.DataPipe

/**
  * Implementation of locally stationary kernels as defined in
  * http://jmlr.csail.mit.edu/papers/volume2/genton01a/genton01a.pdf
  *
  * K(x,y) = K1(x+y/2)&times;K2(x-y)
  *
  * @tparam I The index set or input domain over which the kernel function is evaluated.
  * @param baseKernel The kernel given by K2, it is assumed that the user inputs a valid stationary kernel
  * @param scalingFunc The non-negative scaling function K1(.)
  *
  * */
class LocallyStationaryKernel[I](
  baseKernel: StationaryKernel[I, Double, DenseMatrix[Double]],
  scalingFunc: DataPipe[I, Double])(implicit ev: Field[I])
  extends LocalScalarKernel[I] {

  private val header: String = "LocStatKernel"

  override def toString = {
    val oid = super.toString.split("\\.").last.split("@").last
    header+"@"+oid
  }

  val base_kernel_id = baseKernel.toString.split("\\.").last

  private val stringify = (h: String) => base_kernel_id+"/"+h

  private val revstringify = (h: String) => h.split("/").last

  override val hyper_parameters: List[String] = baseKernel.hyper_parameters.map(stringify)

  blocked_hyper_parameters = baseKernel.blocked_hyper_parameters.map(stringify)

  state = baseKernel.state.map(c => (stringify(c._1), c._2))

  override def block(h: String*) = {
    baseKernel.block(h.map(revstringify):_*)
    blocked_hyper_parameters = h.toList
  }

  override def block_all_hyper_parameters = {
    baseKernel.block_all_hyper_parameters
    blocked_hyper_parameters = hyper_parameters
  }

  override def setHyperParameters(h: Map[String, Double]) = {
    baseKernel.setHyperParameters(h.map(c => (revstringify(c._1), c._2)))
    state = baseKernel.state.map(c => (stringify(c._1), c._2))
    this
  }

  override def evaluateAt(config: Map[String, Double])(x: I, y: I): Double =
    scalingFunc(ev.div(ev.plus(x,y),ev.fromDouble(0.5)))*
      baseKernel.evaluateAt(config.map(c => (revstringify(c._1), c._2)))(x,y)

  override def gradientAt(config: Map[String, Double])(x: I, y: I) =
    baseKernel.gradientAt(config.map(c => (revstringify(c._1), c._2)))(x, y).mapValues(
      _*scalingFunc(ev.div(ev.plus(x,y),ev.fromDouble(0.5)))
    )
}

/**
  *
  * Represents covariance function of a GP which is
  * scaled version of a base GP.
  *
  * z ~ GP(m(.), K(.,.))
  *
  * y = g(x)*z
  *
  * y ~ GP(g(x)*m(x), g(x)K(x,x')g(x'))
  *
  * @author tailhq date: 31/03/2017
  *
  * */
class ScaledKernel[I](
  baseKernel: LocalScalarKernel[I],
  scalingFunc: DataPipe[I, Double])
  extends LocalScalarKernel[I] {

  state = baseKernel.state

  override val hyper_parameters: List[String] = baseKernel.hyper_parameters

  override def evaluateAt(config: Map[String, Double])(x: I, y: I): Double =
    scalingFunc(x)*baseKernel.evaluateAt(config)(x,y)*scalingFunc(y)

  override def gradientAt(config: Map[String, Double])(x: I, y: I) =
    baseKernel.gradientAt(config)(x, y).mapValues(_*scalingFunc(x)*scalingFunc(y))

}

object ScaledKernel {

  /**
    * Create Scaled Kernels on the fly
    * */
  def apply[I](
    baseKernel: LocalScalarKernel[I],
    scalingFunc: DataPipe[I, Double]): ScaledKernel[I] =
    new ScaledKernel(baseKernel, scalingFunc)
}

