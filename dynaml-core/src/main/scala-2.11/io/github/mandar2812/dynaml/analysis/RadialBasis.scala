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
package io.github.mandar2812.dynaml.analysis

import breeze.linalg._
import spire.algebra.{Field, InnerProductSpace, NRoot}
import spire.implicits._
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.utils._

/**
  * <h3>Radial Basis Function Generator<h3>
  *
  * @author mandar2812 date 2017/08/15
  * */
class RadialBasis[I](
  activation: DataPipe[Double, Double])(
  centers: Seq[I], lengthScales: Seq[Double])(
  implicit field: InnerProductSpace[I, Double]) extends Basis[I] {


  override protected val f = (x: I) => DenseVector(
    (
      Seq(1d) ++
      centers.zip(lengthScales).map(cs => {
        val d = field.minus(x, cs._1)
        val r = math.sqrt(field.dot(d, d))/cs._2

        activation(r)
      })
    ).toArray
  )

}

object Radial {

  val gaussianActivation = DataPipe((x: Double) => math.exp(-0.5*x*x))

  val multiquadricActivation = DataPipe((x: Double) => math.sqrt(1d + x*x))

  val invMultiQuadricActivation = DataPipe((x: Double) => 1d/math.sqrt(1d + x*x))

  def gaussianBasis[I](
    centers: Seq[I], lengthScales: Seq[Double])(
    implicit f: InnerProductSpace[I, Double]) =
    new RadialBasis[I](gaussianActivation)(centers, lengthScales)

  def multiquadricBasis[I](
    centers: Seq[I], lengthScales: Seq[Double])(
    implicit f: Field[I] with InnerProductSpace[I, Double]) =
    new RadialBasis[I](multiquadricActivation)(centers, lengthScales)


  def invMultiquadricBasis[I](
    centers: Seq[I], lengthScales: Seq[Double])(
    implicit f: InnerProductSpace[I, Double]) =
    new RadialBasis[I](invMultiQuadricActivation)(centers, lengthScales)

}
