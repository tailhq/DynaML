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

package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * @author mandar2812 date 13/09/16.
  *
  * Implementation of the Maximum Likelihood Perceptron (MLP) kernel
  */
class MLPKernel(w: Double, b: Double) extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable{

  override val hyper_parameters = List("w", "b")

  state = Map("w" -> w, "b" -> b)

  def setw(d: Double): Unit = {
    state += ("w" -> d.toDouble)
  }

  def setoffset(o: Double): Unit = {
    state += ("b" -> o)
  }

  override def evaluateAt(config: Map[String, Double])(
    x: DenseVector[Double],
    y: DenseVector[Double]): Double =
    math.asin(
      (config("w")*(x.t*y) + config("b"))/
      (math.sqrt(config("w")*(x.t*x) + config("b") + 1) * math.sqrt(config("w")*(y.t*y) + config("b") + 1))
    )

  override def gradientAt(config: Map[String, Double])(x: DenseVector[Double], y: DenseVector[Double]) = {
    val (wxy, wxx, wyy) = (
      config("w")*(x.t*y) + config("b"),
      math.sqrt(config("w")*(x.t*x) + config("b") + 1),
      math.sqrt(config("w")*(y.t*y) + config("b") + 1))

    val (numerator, denominator) = (wxy, wxx*wyy)

    val z = numerator/denominator

    val alpha = 1.0/(1.0 - z*z)

    Map(
      "w" ->
        alpha*((denominator*(x.t*y) - numerator*0.5*(wyy*(x.t*x)/wxx + wxx*(y.t*y)/wyy))/(denominator*denominator)),
      "b" ->
        alpha*((denominator - numerator*0.5*(wyy/wxx + wxx/wyy))/(denominator*denominator))
    )
  }
}
