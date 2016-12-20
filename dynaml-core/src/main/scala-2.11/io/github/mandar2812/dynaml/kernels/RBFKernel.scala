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
import spire.algebra.{Field, InnerProductSpace}

/**
 * RBF Kernels of the form
 * K(x,y) = exp(-||x - y||<sup>2</sup>/2 &#215; l<sup>2</sup>)
 */

class GenericRBFKernel[T](private var bandwidth: Double = 1.0)(
  implicit ev: Field[T] with InnerProductSpace[T, Double])
  extends StationaryKernel[T, Double, DenseMatrix[Double]]
    with LocalScalarKernel[T] with Serializable { self =>
  override val hyper_parameters = List("bandwidth")

  state = Map("bandwidth" -> bandwidth)

  def setbandwidth(d: Double): Unit = {
    this.state += ("bandwidth" -> d)
    this.bandwidth = d
  }

  override def eval(x: T): Double =
    math.exp(-1*ev.dot(x, x)/(2*math.pow(this.state("bandwidth"), 2)))

  override def gradient(x: T,
                        y: T): Map[String, Double] =
    Map("bandwidth" -> 1.0*evaluate(x,y)*math.pow(ev.dot(x, x),2)/math.pow(math.abs(state("bandwidth")), 3))

  def getBandwidth: Double = this.bandwidth

  def >[K <: LocalScalarKernel[T]](otherKernel: K): CompositeCovariance[T] = {

    new CompositeCovariance[T] {
      override val hyper_parameters = self.hyper_parameters ++ otherKernel.hyper_parameters

      override def evaluate(x: T, y: T) = {
        val arg = otherKernel.evaluate(x,y) +
          otherKernel.evaluate(y,y) -
          2.0*otherKernel.evaluate(x,y)

        math.exp(-1.0*arg/(2.0*math.pow(state("bandwidth"), 2.0)))
      }

      state = self.state ++ otherKernel.state

      override def gradient(x: T, y: T): Map[String, Double] = {
        val arg = otherKernel.evaluate(x,y) +
          otherKernel.evaluate(y,y) -
          2.0*otherKernel.evaluate(x,y)

        val gradx = otherKernel.gradient(x,x)
        val grady = otherKernel.gradient(y,y)
        val gradxy = otherKernel.gradient(x,y)

        Map("bandwidth" ->
          this.evaluate(x,y)*arg/math.pow(math.abs(state("bandwidth")), 3)
        ) ++
          gradxy.map((s) => {
            val ans = (-2.0*s._2 + gradx(s._1) + grady(s._1))/2.0*math.pow(state("bandwidth"), 2.0)
            (s._1, -1.0*this.evaluate(x,y)*ans)
          })
      }

      override def buildKernelMatrix[S <: Seq[T]](mappedData: S, length: Int) =
        SVMKernel.buildSVMKernelMatrix(mappedData, length, this.evaluate)

      override def buildCrossKernelMatrix[S <: Seq[T]](dataset1: S, dataset2: S) =
        SVMKernel.crossKernelMatrix(dataset1, dataset2, this.evaluate)

    }
  }

}

class RBFKernel(private var bandwidth: Double = 1.0)(
  implicit ev: Field[DenseVector[Double]] with InnerProductSpace[DenseVector[Double], Double])
  extends GenericRBFKernel[DenseVector[Double]](bandwidth)(ev)
  with SVMKernel[DenseMatrix[Double]]
  with Serializable { self =>

  //val normedVectorSpace = ev.normed


}

/**
  * Squared Exponential Kernel is a generalized RBF Kernel
  * K(x,y) = h<sup>2</sup>*exp(-||x - y||<sup>2</sup>/2 &#215; l<sup>2</sup>)
  */
class SEKernel(private var band: Double = 1.0, private var h: Double = 2.0)(
  implicit ev: Field[DenseVector[Double]] with InnerProductSpace[DenseVector[Double], Double])
  extends RBFKernel(band) {

  state = Map("bandwidth" -> band, "amplitude" -> h)

  override val hyper_parameters = List("bandwidth","amplitude")

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]) =
    math.pow(state("amplitude"), 2.0)*super.evaluate(x,y)

  override def gradient(x: DenseVector[Double],
                        y: DenseVector[Double]): Map[String, Double] =
    Map("amplitude" -> 2.0*state("amplitude")*super.evaluate(x,y)) ++ super.gradient(x,y)

}

/**
  * Mahalanobis kernel is an anisotropic generalization of the
  * RBF Kernel, its definition is based on the so called Mahalanobis
  * distance between two vectors x and y.
  *
  * K(x,y) = h*exp(-(x - y)<sup>T</sup> . M . (x - y))
  *
  * In this implementation the symmetric
  * positive semi-definite matrix M is assumed to
  * be diagonal.
  */
class MahalanobisKernel(private var band: DenseVector[Double], private var h: Double = 2.0)
  extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable  {

  state = Map("MahalanobisAmplitude" -> h) ++
    band.mapPairs((i, b) => ("MahalanobisBandwidth_"+i, b)).toArray.toMap

  override val hyper_parameters = List("MahalanobisAmplitude") ++
    band.mapPairs((i, b) => "MahalanobisBandwidth_"+i).toArray.toList

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]) = {
    val bandMap = state.filter((k) => k._1.contains("MahalanobisBandwidth"))
    assert(x.length == bandMap.size,
      "Mahalanobis Bandwidth vector's must be equal to that of data: "+x.length)
    val diff = x - y
    val bandwidth = DenseMatrix.tabulate[Double](bandMap.size, bandMap.size)((i, j) => {
      if (i == j)
        math.pow(bandMap("MahalanobisBandwidth_"+i), -2.0)
      else
        0.0
    })

    math.pow(state("MahalanobisAmplitude"), 2.0)*
      math.exp((diff.t*(bandwidth*diff))*(-1.0/2.0))
  }

  override def gradient(x: DenseVector[Double],
                        y: DenseVector[Double]): Map[String, Double] = {
    val bandMap = state.filter((k) => k._1.contains("MahalanobisBandwidth"))
    assert(x.length == bandMap.size, "Mahalanobis Bandwidth vector's must be equal to that of data")
    Map("MahalanobisAmplitude" -> 2.0*evaluate(x,y)/state("MahalanobisAmplitude")) ++
      bandMap.map((k) => (k._1, evaluate(x,y)*2.0/math.pow(k._2, 3.0)))
  }

}


class RBFCovFunc(private var bandwidth: Double)
  extends LocalSVMKernel[Double] {
  override val hyper_parameters: List[String] = List("bandwidth")

  state = Map("bandwidth" -> bandwidth)

  override def evaluate(x: Double, y: Double): Double = {
    val diff = x - y
    Math.exp(-1*math.pow(diff, 2)/(2*math.pow(this.state("bandwidth"), 2)))
  }

  override def gradient(x: Double, y: Double): Map[String, Double] =
    Map("bandwidth" -> evaluate(x,y)*math.pow(x-y,2)/math.pow(math.abs(state("bandwidth")), 3))
}

class SECovFunc(private var band: Double = 1.0, private var h: Double = 2.0)
  extends RBFCovFunc(band) {

  state = Map("bandwidth" -> band, "amplitude" -> h)

  override val hyper_parameters = List("bandwidth","amplitude")

  override def evaluate(x: Double, y: Double) =
    math.pow(state("amplitude"), 2.0)*super.evaluate(x,y)

  override def gradient(x: Double,
                        y: Double): Map[String, Double] =
    Map("amplitude" -> 2.0*state("amplitude")*super.evaluate(x,y)) ++ super.gradient(x,y)

}

class CoRegRBFKernel(bandwidth: Double) extends LocalSVMKernel[Int] {

  override val hyper_parameters: List[String] = List("coRegB")

  state = Map("coRegB" -> bandwidth)

  override def gradient(x: Int, y: Int): Map[String, Double] =
    Map("coRegB" -> 1.0*evaluate(x,y)*math.pow(x-y,2)/math.pow(math.abs(state("coRegB")), 3))

  override def evaluate(x: Int, y: Int): Double = {
    math.exp(-1.0*math.pow(x-y, 2.0)/state("coRegB"))
  }
}
