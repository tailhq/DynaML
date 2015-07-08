/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{DenseVector, norm}
import io.github.mandar2812.dynaml.utils

class GaussianDensityKernel
  extends DensityKernel
  with Serializable {
  private val exp = scala.math.exp _
  private val pow = scala.math.pow _
  private val sqrt = scala.math.sqrt _
  private val Pi = scala.math.Pi
  protected var bandwidth: DenseVector[Double] = DenseVector.zeros(10)
  override protected val mu = (1/4)*(1/sqrt(Pi))
  override protected val r = (1/2)*(1/sqrt(Pi))

  private def evalForDimension(x: Double, pilot: Double): Double =
    exp(-1*pow(x/pilot, 2)/2)/sqrt(Pi * 2)

  private def evalWithBandwidth(x: DenseVector[Double], b: DenseVector[Double]): Double = {
    assert(x.size == b.size,
      "Dimensions of vector x and the bandwidth of the kernel must match")
    val buff = x
    val bw = b
    val normalizedbuff: breeze.linalg.DenseVector[Double] = DenseVector.tabulate(
      bw.size)(
        (i) => buff(i)/bw(i)
      )
    exp(-1*pow(norm(normalizedbuff), 2)/2)/pow(sqrt(Pi * 2), b.size)
  }

  def setBandwidth(b: DenseVector[Double]): Unit = {
    this.bandwidth = b
  }

  override def eval(x: DenseVector[Double]) = evalWithBandwidth(x, this.bandwidth)

  /**
   * Calculates the derivative at point x for the Gaussian
   * Density Kernel, for only one dimension.
   *
   * @param n The number of times the gaussian has to be differentiated
   * @param x The point x at which the derivative has to evaluated
   * @return The value of the nth derivative at the point x
   * */
  override def derivative(n: Int, x: Double): Double = {
    (1/sqrt(2*Pi))*(1/pow(-1.0,n))*exp(-1*pow(x,2)/2)*utils.hermite(n, x)
  }

}
