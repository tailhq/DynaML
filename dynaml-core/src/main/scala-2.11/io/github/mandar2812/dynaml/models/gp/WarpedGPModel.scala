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
package io.github.mandar2812.dynaml.models.gp

import breeze.linalg.{DenseMatrix, det, diag}
import io.github.mandar2812.dynaml.DynaMLPipe.identityPipe
import io.github.mandar2812.dynaml.algebra.{PartitionedMatrix, PartitionedVector}
import io.github.mandar2812.dynaml.analysis.{DifferentiableMap, PartitionedVectorField, PushforwardMap}
import io.github.mandar2812.dynaml.models.{ContinuousProcessModel, SecondOrderProcessModel}
import io.github.mandar2812.dynaml.optimization.GloballyOptWithGrad
import io.github.mandar2812.dynaml.pipes.{DataPipe, Encoder}
import io.github.mandar2812.dynaml.probability.distributions.BlockedMultiVariateGaussian
import io.github.mandar2812.dynaml.probability.{E, MeasurableDistrRV}
import io.github.mandar2812.dynaml.utils
import org.apache.spark.annotation.Experimental
import spire.algebra.Field
import spire.implicits._

import scala.reflect.ClassTag

/**
  * ::Experimental::
  *
  * A warped Gaussian Process.
  * @author mandar date 02/01/2017.
  *
  * */
@Experimental
class WarpedGPModel[T, I:ClassTag](
  p: AbstractGPRegressionModel[T, I])(
  val warpingFunc: PushforwardMap[Double, Double, Double])(
  implicit pf: PartitionedVectorField,
  transform: Encoder[T, Seq[(I, Double)]])
  extends ContinuousProcessModel[
    T, I, Double,
    MeasurableDistrRV[PartitionedVector, PartitionedVector, PartitionedMatrix, BlockedMultiVariateGaussian]]
  with SecondOrderProcessModel[
    T, I, Double, Double, DenseMatrix[Double],
    MeasurableDistrRV[PartitionedVector, PartitionedVector, PartitionedMatrix, BlockedMultiVariateGaussian]]
  with GloballyOptWithGrad {

  /**
    * The training data
    **/
  override protected val g: T = p.data

  private val dataProcessPipe = transform >
    DataPipe((s: Seq[(I, Double)]) => s.map(pattern => (pattern._1, warpingFunc.i(pattern._2)))) >
    transform.i

  val underlyingProcess =
    AbstractGPRegressionModel[T, I](
      p.covariance, p.noiseModel, p.mean)(
      dataProcessPipe(p.data), p.npoints)


  /**
    * Mean Function: Takes a member of the index set (input)
    * and returns the corresponding mean of the distribution
    * corresponding to input.
    **/
  override val mean = p.mean > warpingFunc

  /**
    * Underlying covariance function of the
    * Gaussian Processes.
    **/
  override val covariance = p.covariance

  /**
    * Stores the names of the hyper-parameters
    **/
  override protected var hyper_parameters: List[String] = underlyingProcess._hyper_parameters

  /**
    * A Map which stores the current state of
    * the system.
    **/
  override protected var current_state: Map[String, Double] = underlyingProcess._current_state

  //Define the default determinant implementation
  implicit val detImpl = DataPipe(
    (m: PartitionedMatrix) => m.filterBlocks(c => c._1 == c._2).map(c => det(c._2)).product)

  //Define the push forward map for the multivariate case
  val wFuncPredDistr: PushforwardMap[PartitionedVector, PartitionedVector, PartitionedMatrix] =
    PushforwardMap(
      DataPipe((v: PartitionedVector) => v.map(c => (c._1, c._2.map(warpingFunc.run)))),
      DifferentiableMap(
        (v: PartitionedVector) => v.map(c => (c._1, c._2.map(warpingFunc.i.run))),
        (v: PartitionedVector) => new PartitionedMatrix(
          v._data.map(l => ((l._1, l._1), diag(l._2.map(warpingFunc.i.J)))) ++
            utils.combine(Seq((0 until v.rowBlocks.toInt).toList, (0 until v.rowBlocks.toInt).toList))
              .map(c =>
                (c.head.toLong, c.last.toLong))
              .filter(c => c._2 != c._1)
              .map(c => (c, DenseMatrix.zeros[Double](v.rows.toInt/v.rowBlocks.toInt, v.rows.toInt/v.rowBlocks.toInt)))
              .toStream, num_cols = v.rows, num_rows = v.rows))
      )

  /**
    * Draw three predictions from the posterior predictive distribution
    * 1) Mean or MAP estimate Y
    * 2) Y- : The lower error bar estimate (mean - sigma*stdDeviation)
    * 3) Y+ : The upper error bar. (mean + sigma*stdDeviation)
    **/
  override def predictionWithErrorBars[U <: Seq[I]](testData: U, sigma: Int) =
    underlyingProcess
      .predictionWithErrorBars(testData, sigma)
      .map(d => (d._1, warpingFunc(d._2), warpingFunc(d._3), warpingFunc(d._4)))

  /**
    * Calculates the energy of the configuration,
    * in most global optimization algorithms
    * we aim to find an approximate value of
    * the hyper-parameters such that this function
    * is minimized.
    *
    * @param h       The value of the hyper-parameters in the configuration space
    * @param options Optional parameters about configuration
    * @return Configuration Energy E(h)
    **/
  override def energy(h: Map[String, Double], options: Map[String, String]) = {
    val trainingLabels = PartitionedVector(
      dataAsSeq(g).toStream.map(_._2),
      underlyingProcess.npoints.toLong, underlyingProcess._blockSize
    )

    detImpl(wFuncPredDistr.i.J(trainingLabels))*underlyingProcess.energy(h, options)
  }


  /** Calculates posterior predictive distribution for
    * a particular set of test data points.
    *
    * @param test A Sequence or Sequence like data structure
    *             storing the values of the input patters.
    **/
  override def predictiveDistribution[U <: Seq[I]](test: U) =
    wFuncPredDistr -> underlyingProcess.predictiveDistribution(test)

  /**
    * Convert from the underlying data structure to
    * Seq[(I, Y)] where I is the index set of the GP
    * and Y is the value/label type.
    **/
  override def dataAsSeq(data: T) = transform(data)

  /**
    * Predict the value of the
    * target variable given a
    * point.
    *
    **/
  override def predict(point: I) = warpingFunc(underlyingProcess.predictionWithErrorBars(Seq(point), 1).head._2)
}

/**
  * <h3>Log Gaussian Processes</h3>
  *
  * A stochastic process whose logarithm is a gaussian process.
  *
  * @author mandar2812 date 27/07/17
  * */
class LogGaussianProcess[T, I:ClassTag](p: AbstractGPRegressionModel[T, I])(
  implicit pf: PartitionedVectorField,
  transform: Encoder[T, Seq[(I, Double)]]) extends
  WarpedGPModel[T, I](p)(
    PushforwardMap(
      DataPipe((x: Double) => math.exp(x)),
      DifferentiableMap(
        (x: Double) => math.log(x),
        (x: Double) => 1.0/x))(
      identityPipe[Double], Field[Double])
  )