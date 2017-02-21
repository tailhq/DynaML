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
package io.github.mandar2812.dynaml.models.bayes

import spire.algebra.{Field, InnerProductSpace}
import io.github.mandar2812.dynaml.algebra.PartitionedVector
import io.github.mandar2812.dynaml.analysis.PartitionedVectorField
import io.github.mandar2812.dynaml.kernels.LocalScalarKernel
import io.github.mandar2812.dynaml.modelpipe.GPRegressionPipe2
import io.github.mandar2812.dynaml.models.gp.AbstractGPRegressionModel

import scala.reflect.ClassTag
import io.github.mandar2812.dynaml.pipes.MetaPipe
import io.github.mandar2812.dynaml.probability.MultGaussianPRV
import org.apache.spark.annotation.Experimental

/**
  * @author mandar2812 date: 21/02/2017.
  */
@Experimental
abstract class AbstractGaussianProcessPrior[I: ClassTag, MeanFuncParams](
  covariance: LocalScalarKernel[I],
  noiseCovariance: LocalScalarKernel[I]) extends
  StochasticProcessPrior[
    I, Double, PartitionedVector,
    MultGaussianPRV, MultGaussianPRV,
    AbstractGPRegressionModel[Seq[(I, Double)], I]] {

  def _meanFuncParams: MeanFuncParams

  def meanFuncParams_(p: MeanFuncParams): Unit

  val meanFunctionPipe: MetaPipe[MeanFuncParams, I, Double]

  val posteriorModelPipe = new GPRegressionPipe2[I](covariance, noiseCovariance)


  override def posteriorModel(data: Seq[(I, Double)]) =
    posteriorModelPipe(data, meanFunctionPipe(_meanFuncParams))

  override def priorDistribution[U <: Seq[I]](d: U) = {

    val numPoints: Long = d.length.toLong

    //Declare vector field, required implicit parameter
    implicit val field: Field[PartitionedVector] =
      PartitionedVectorField(numPoints, covariance.rowBlocking)

    //Construct mean Vector
    val meanFunc = meanFunctionPipe(_meanFuncParams)
    val meanVector = PartitionedVector(
      d.toStream.map(meanFunc(_)),
      numPoints,
      covariance.rowBlocking)

    //Construct covariance matrix
    val covMat = covariance.buildBlockedKernelMatrix(d, numPoints)

    MultGaussianPRV(meanVector, covMat)
  }
}


class LinearTrendGaussianPrior[I: ClassTag](
  covariance: LocalScalarKernel[I],
  noiseCovariance: LocalScalarKernel[I],
  trendParams: I)(
  implicit inner: InnerProductSpace[I, Double]) extends
  AbstractGaussianProcessPrior[I, I](covariance, noiseCovariance) {

  private var params = trendParams

  override def _meanFuncParams = params

  override def meanFuncParams_(p: I) = params = p

  override val meanFunctionPipe = MetaPipe((params: I) => (x: I) => inner.dot(params, x))
}
