/*
Copyright 2015 Mandar Chandorkar

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
package io.github.mandar2812.dynaml.models.stp

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.kernels.{LocalScalarKernel, SVMKernel}
import io.github.mandar2812.dynaml.models.ContinuousProcessModel
import io.github.mandar2812.dynaml.optimization.GloballyOptimizable
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.probability.MatrixTRV
import org.apache.log4j.Logger

import scala.reflect.ClassTag

/**
  * Implementation of Conti & O' Hagan Bayesian
  * multi-output Students' T model which
  * can be used in regression and simulator
  * emulation.
  *
  * Reference:
  * <a href="http://www.sciencedirect.com/science/article/pii/S0378375809002559">
  *   Journal of Statistical Planning and Inference
  * </a>
  *
  * @tparam T The type of the training data features and outputs.
  * @tparam I The type of the index set (input features) of the process.
  * @param cov The covariance function of the output function.
  * @param n The covariance of the measurement noise.
  * @param h The trend function/feature map as a DynaML [[DataPipe]].
  * @author mandar2812 date 28/04/2017.
  * */
abstract class MVStudentsTModel[T, I: ClassTag](
  cov: LocalScalarKernel[I], n: LocalScalarKernel[I],
  data: T, num: Int, numOutputs: Int,
  h: DataPipe[I, DenseVector[Double]]) extends
  ContinuousProcessModel[T, I, DenseVector[Double], MatrixTRV] with
  GloballyOptimizable {

  private val logger = Logger.getLogger(this.getClass)

  val npoints = num

  val num_outputs = numOutputs

  val featureMap: DataPipe[I, DenseVector[Double]] = h

  protected lazy val (trainingData, trainingDataLabels): (Seq[I], Seq[DenseVector[Double]]) = dataAsSeq(g).unzip

  val num_latent_features = featureMap(trainingData.head).length

  require(
    npoints >= num_latent_features + num_outputs,
    "In a Baysian Multi-output Students T model, n >= m+q")

  protected lazy val H: DenseMatrix[Double] = DenseMatrix.vertcat(trainingData.map(featureMap(_).asDenseMatrix):_*)

  protected lazy val D: DenseMatrix[Double] = DenseMatrix.vertcat(trainingDataLabels.map(_.asDenseMatrix):_*)

  protected var (caching, kernelMatrixCache, bGLSCache, sigmaGLSCache, residualGLSCache, hAhCache)
  : (Boolean, DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double]) =
    (false, null, null, null, null, null)


  /**
    * Set the model "state" which
    * contains values of its hyper-parameters
    * with respect to the covariance and noise
    * kernels.
    * */
  def setState(s: Map[String, Double]): this.type = {

    val (covHyp, noiseHyp) = (
      s.filterKeys(covariance.hyper_parameters.contains),
      s.filterKeys(noiseModel.hyper_parameters.contains)
    )

    covariance.setHyperParameters(covHyp)
    noiseModel.setHyperParameters(noiseHyp)
    current_state = covariance.state ++ noiseModel.state
    this
  }

  /**
    * Underlying covariance function of the
    * Gaussian Processes.
    **/
  val covariance = cov

  val noiseModel = n

  /**
    * Stores the names of the hyper-parameters
    **/
  override protected var hyper_parameters: List[String] = covariance.hyper_parameters ++ noiseModel.hyper_parameters
  /**
    * A Map which stores the current state of
    * the system.
    **/
  override protected var current_state: Map[String, Double] = covariance.state ++ noiseModel.state

  /** Calculates posterior predictive distribution for
    * a particular set of test data points.
    *
    * @param test A Sequence or Sequence like data structure
    *             storing the values of the input patters.
    **/
  override def predictiveDistribution[U <: Seq[I]](test: U) = {

    val (kMat, bGLS, sigmaGLS, resGLS, hAh) = if(caching) {
      logger.info("** Using cached training matrix **")
      (kernelMatrixCache, bGLSCache, sigmaGLSCache, residualGLSCache, hAhCache)
    } else {
      val effectiveTrainingKernel: LocalScalarKernel[I] = this.covariance + this.noiseModel

      logger.info("---------------------------------------------------------------")
      logger.info("Calculating covariance matrix for training points")
      val A = SVMKernel.buildSVMKernelMatrix(
        trainingData, trainingData.length,
        effectiveTrainingKernel.evaluate)
        .getKernelMatrix()

      val (abyH, abyD) = (A\H, A\D)

      val h_A_h = H.t*abyH
      val b_hat = h_A_h\(H.t*abyD)

      val residual_gls = D - H*b_hat

      val s_hat = residual_gls.t*(A\residual_gls)
      (A, b_hat, s_hat.mapValues(v => v/(npoints-num_latent_features).toDouble), residual_gls, h_A_h)
    }

    logger.info("---------------------------------------------------------------")
    logger.info("Calculating covariance matrix for test points")
    val kernelTest = SVMKernel.buildSVMKernelMatrix(
      test, test.length, covariance.evaluate).getKernelMatrix()

    logger.info("---------------------------------------------------------------")
    logger.info("Calculating covariance matrix between training and test points")
    val crossKernel = SVMKernel.crossKernelMatrix(
      trainingData, test, covariance.evaluate)

    val aByT = kMat\crossKernel

    val hTest = DenseMatrix.vertcat(test.map(featureMap(_).asDenseMatrix):_*)
    val hTestRes = hTest - H.t*aByT

    val predictiveMean = bGLS.t*H + resGLS.t*aByT

    val predictiveCovariance = kernelTest - crossKernel.t*aByT + hTestRes.t*(hAh\hTestRes)
    MatrixTRV(npoints - num_latent_features, predictiveMean, predictiveCovariance, sigmaGLS)
  }

  /**
    * The training data
    **/
  override protected val g: T = data

  /**
    * Predict the value of the
    * target variable given a
    * point.
    *
    **/
  override def predict(point: I) = ???
}
