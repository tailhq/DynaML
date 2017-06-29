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
import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.pipes.Encoder
import spire.algebra.NormedVectorSpace

/**
  * <h3>Cubic Spline Covariance</h3>
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

/**
  * <h3> Cubic Spline Kernel (ARD)</h3>
  *
  * <i>Automatic relevance determination</i> (ARD) version of the
  * cubic spline kernel.
  *
  * @tparam I Input domain
  * @param theta Length scales
  * @param enc An [[Encoder]] instance which converts between
  *            hyper-parameter configurations specified as [[Map]]
  *            and the generic type [[I]]
  * @param f Implicitly specified [[Field]] over the type [[I]],
  *          enables element-wise algebra
  * @param n Implicitly specified [[NormedVectorSpace]] over the type [[I]],
  *          enables calculation of norms.
  * */
abstract class CubicSplineARDKernel[I](
  theta: I, enc: Encoder[Map[String, Double], I])(
  implicit f: Field[I], n: NormedVectorSpace[I, Double]) extends
  StationaryKernel[I, Double, DenseMatrix[Double]] with
  LocalScalarKernel[I] {

  val parameter_encoder: Encoder[Map[String, Double], I] = enc

  state = parameter_encoder.i(theta)

  override val hyper_parameters = state.keys

  override def evalAt(config: Map[String, Double])(x: I) = {
    val d = n.norm(x)
    val th = parameter_encoder(config)
    val dth = n.norm(f.div(x, th))

    if(dth < 0.5) 1 - 6d*dth*dth + 6d*dth*dth*dth
    else if(dth >= 0.5 && d < 1d) 2d*math.pow(1d - dth, 3d)
    else 0d
  }

  /*
  override def gradientAt(config: Map[String, Double])(x: I, y: I) = {
    val d = f.minus(x,y)
    val th = parameter_encoder(config)
    val dth = n.norm(f.div(d, th))

    state.map(th => {
      val (t, v) = th
      (t,
        if(dth < 0.5) 12d*dth*d/math.pow(th, 3) - 18d*d*d*d/math.pow(th, 4)
        else if(dth >= 0.5 && d < 1d) -6d*math.pow(1d - dth, 2d)*d
        else 0d)
    })
  }*/
}

object CubicSplineARDKernel {

  def getEncoderforBreezeDV(dim: Int): Encoder[Map[String, Double], DenseVector[Double]] =
    Encoder(
      (conf: Map[String, Double]) => DenseVector((0 until dim).map(i => conf("theta_"+i)).toArray),
      (c: DenseVector[Double]) => c.mapPairs((i, v) => ("theta_"+i, v)).toArray.toMap
    )

}