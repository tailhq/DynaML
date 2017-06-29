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

import spire.algebra.Field
import breeze.linalg.DenseMatrix
import spire.algebra.NormedVectorSpace

/**
  * <h3>Cubic Spline Interpolation Covariance</h3>
  *
  * Implementation of the <i>cubic spline</i> kernel/covariance function,
  * for arbitrary domains [[I]] over which a field and norm are defined as implicits.
  *
  * @tparam I The index set/domain over which the kernel is defined
  * @param theta The value of the length scale &theta;
  * @author mandar2812 date 29/06/2017.
  * */
class CubicSplineKernel[I](theta: Double)(
  implicit f: Field[I], n: NormedVectorSpace[I, Double]) extends
  StationaryKernel[I, Double, DenseMatrix[Double]] with
  LocalScalarKernel[I] {

  override def evalAt(config: Map[String, Double])(x: I) = {
    val d = n.norm(x)
    val th = config("theta")
    val dth = d/th

    if(d < th/2d) 1 - 6d*dth*dth + 6d*dth*dth*dth
    else if(d >= th/2d && d < th) 2d*math.pow(1d - dth, 3d)
    else 0d
  }

  override val hyper_parameters = List("theta")

  state = Map("theta" -> theta)

  override def gradientAt(config: Map[String, Double])(x: I, y: I) = {
    val d = n.norm(f.minus(x,y))
    val th = config("theta")
    val dth = d/th

    Map("theta" -> {
      if(d < th/2d) 12d*d*d/math.pow(th, 3) - 18d*d*d*d/math.pow(th, 4)
      else if(d >= th/2d && d < th) -6d*math.pow(1d - dth, 2d)*d
      else 0d
    })
  }
}
