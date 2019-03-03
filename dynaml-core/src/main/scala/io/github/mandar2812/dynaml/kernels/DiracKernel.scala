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

import breeze.linalg.{DenseMatrix, DenseVector, norm}

/**
  * Dirac kernel is equivalent to the
  * classical Dirac delta function scaled by
  * a hyper-parameter called the noise level.
  *
  * K(x,y) = noise*DiracDelta(x,y)
  */
class DiracKernel(private var noiseLevel: Double = 1.0)
  extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable {

  override val hyper_parameters = List("noiseLevel")

  state = Map("noiseLevel" -> noiseLevel)

  def setNoiseLevel(d: Double): Unit = {
    this.state += ("noiseLevel" -> d)
    this.noiseLevel = d
  }

  override def evaluateAt(
    config: Map[String, Double])(
    x: DenseVector[Double],
    y: DenseVector[Double]): Double =
    if (norm(x-y, 2) == 0) math.abs(config("noiseLevel"))*1.0 else 0.0

  override def gradientAt(
    config: Map[String, Double])(
    x: DenseVector[Double],
    y: DenseVector[Double]): Map[String, Double] =
    Map("noiseLevel" -> 1.0*evaluateAt(config)(x,y)/math.abs(config("noiseLevel")))

  override def buildKernelMatrix[S <: Seq[DenseVector[Double]]](mappedData: S,
                                                                length: Int)
  : KernelMatrix[DenseMatrix[Double]] =
    new SVMKernelMatrix(DenseMatrix.eye[Double](length)*state("noiseLevel"), length)

}

class MAKernel(private var noiseLevel: Double = 1.0)
  extends LocalSVMKernel[Double]
  with Serializable {
  override val hyper_parameters = List("noiseLevel")

  state = Map("noiseLevel" -> noiseLevel)

  def setNoiseLevel(d: Double): Unit = {
    this.state += ("noiseLevel" -> d)
    this.noiseLevel = d
  }

  override def evaluateAt(
    config: Map[String, Double])(
    x: Double,
    y: Double): Double =
    if (x-y == 0.0) math.abs(config("noiseLevel"))*1.0 else 0.0

  override def gradientAt(
    config: Map[String, Double])(
    x: Double, y: Double): Map[String, Double] =
    Map("noiseLevel" -> 1.0*evaluateAt(config)(x,y)/math.abs(config("noiseLevel")))

  override def buildKernelMatrix[S <: Seq[Double]](mappedData: S,
                                                   length: Int)
  : KernelMatrix[DenseMatrix[Double]] =
    new SVMKernelMatrix(DenseMatrix.eye[Double](length)*state("noiseLevel"), length)

}

class CoRegDiracKernel extends LocalSVMKernel[Int] {
  override val hyper_parameters: List[String] = List()

  override def gradientAt(config: Map[String, Double])(x: Int, y: Int): Map[String, Double] = Map()

  override def evaluateAt(config: Map[String, Double])(x: Int, y: Int): Double =
    if(x == y) 1.0 else 0.0
}
