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
import spire.algebra.InnerProductSpace
import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.utils._

/**
  * <h3>Radial Basis Function Generator<h3>
  *
  * @author mandar2812 date 2017/08/15
  * */
class RadialBasis[I](
  activation: DataPipe[Double, Double],
  bias: Boolean = true)(
  val centers: Seq[I], val lengthScales: Seq[Double])(
  implicit field: InnerProductSpace[I, Double]) extends Basis[I] {


  override protected val f: (I) => DenseVector[Double] = (x: I) => DenseVector(
    if(bias) {
      (
        Seq(1d) ++
          centers.zip(lengthScales).map(cs => {
            val d = field.minus(x, cs._1)
            val r = math.sqrt(field.dot(d, d))/cs._2

            activation(r)
          })
        ).toArray
    } else {
      centers.zip(lengthScales).map(cs => {
        val d = field.minus(x, cs._1)
        val r = math.sqrt(field.dot(d, d))/cs._2

        activation(r)
      }).toArray
    }
  )

}

object RadialBasis {

  val gaussian = DataPipe((x: Double) => math.exp(-0.5*x*x))

  val laplacian = DataPipe((x: Double) => math.exp(-0.5*math.abs(x)))

  val multiquadric = DataPipe((x: Double) => math.sqrt(1d + x*x))

  val invMultiQuadric = DataPipe((x: Double) => 1d/math.sqrt(1d + x*x))

  val maternHalfInteger = MetaPipe((p: Int) => (x: Double) => {
    //Calculate the matern half integer expression

    //Constants
    val n = factorial(p).toDouble/factorial(2*p)
    val adj_nu = math.sqrt(2*p+1d)
    //Exponential Component
    val ex = math.exp(-1d*adj_nu*x)
    //Polynomial Component
    val poly = (0 to p).map(i => {
      math.pow(adj_nu*2*x, p-i)*factorial(p+i)/(factorial(i)*factorial(p-i))
    }).sum

    poly*ex*n
  })

  def gaussianBasis[I](
    centers: Seq[I],
    lengthScales: Seq[Double],
    bias: Boolean = true)(
    implicit f: InnerProductSpace[I, Double]) =
    new RadialBasis[I](gaussian, bias)(centers, lengthScales)

  def laplacianBasis[I](
    centers: Seq[I],
    lengthScales: Seq[Double],
    bias: Boolean = true)(
    implicit f: InnerProductSpace[I, Double]) =
    new RadialBasis[I](laplacian, bias)(centers, lengthScales)

  def multiquadricBasis[I](
    centers: Seq[I],
    lengthScales: Seq[Double],
    bias: Boolean = true)(
    implicit f: InnerProductSpace[I, Double]) =
    new RadialBasis[I](multiquadric, bias)(centers, lengthScales)

  def invMultiquadricBasis[I](
    centers: Seq[I],
    lengthScales: Seq[Double],
    bias: Boolean = true)(
    implicit f: InnerProductSpace[I, Double]) =
    new RadialBasis[I](invMultiQuadric, bias)(centers, lengthScales)

  def maternHalfIntegerBasis[I](p: Int)(
    centers: Seq[I], lengthScales: Seq[Double],
    bias: Boolean = true)(
    implicit f: InnerProductSpace[I, Double]) =
    new RadialBasis[I](maternHalfInteger(p), bias)(centers, lengthScales)


}
