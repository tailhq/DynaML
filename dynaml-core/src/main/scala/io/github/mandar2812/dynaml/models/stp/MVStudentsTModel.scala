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

import breeze.linalg.{DenseMatrix, DenseVector, det}
import breeze.numerics.log
import io.github.mandar2812.dynaml.kernels.{LocalScalarKernel, SVMKernel}
import io.github.mandar2812.dynaml.models.{ContinuousProcessModel, SecondOrderProcessModel}
import io.github.mandar2812.dynaml.models.stp.MVStudentsTModel.InferencePrimitives
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
  SecondOrderProcessModel[T, I, DenseVector[Double], Double, DenseMatrix[Double], MatrixTRV] with
  GloballyOptimizable {

  private val logger = Logger.getLogger(this.getClass)

  /**
    * The training data
    * */
  override protected val g: T = data

  val npoints = num

  val num_outputs = numOutputs

  val featureMap: DataPipe[I, DenseVector[Double]] = h

  protected lazy val (trainingData, trainingDataLabels): (Seq[I], Seq[DenseVector[Double]]) = dataAsSeq(g).unzip

  val num_latent_features = featureMap(trainingData.head).length

  require(
    npoints >= num_latent_features + num_outputs,
    "In a Bayesian Multi-output Students T model, n >= m+q")

  protected lazy val H: DenseMatrix[Double] = DenseMatrix.vertcat(trainingData.map(featureMap(_).asDenseMatrix):_*)

  protected lazy val D: DenseMatrix[Double] = DenseMatrix.vertcat(trainingDataLabels.map(_.asDenseMatrix):_*)

  protected var caching: Boolean = false

  protected var (kernelMatrixCache, bGLSCache, sigmaGLSCache, residualGLSCache, hAhCache): InferencePrimitives =
    (null, null, null, null, null)


  /**
    * Mean Function: Takes a member of the index set (input)
    * and returns the corresponding mean of the distribution
    * corresponding to input.
    **/
  override val mean = DataPipe((_: I) => DenseVector.zeros[Double](num_outputs))

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
  override val covariance = cov

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
    * */
  override def predictiveDistribution[U <: Seq[I]](test: U) = {

    val (kMat, bGLS, sigmaGLS, resGLS, hAh) = if(caching) {
      println("** Using cached training matrix **")
      (kernelMatrixCache, bGLSCache, sigmaGLSCache, residualGLSCache, hAhCache)
    } else MVStudentsTModel.inferencePrimitives(covariance, noiseModel, trainingData, H, D)

    println("---------------------------------------------------------------")
    println("Calculating covariance matrix for test points")
    val kernelTest = SVMKernel.buildSVMKernelMatrix(
      test, test.length,
      covariance.evaluate)
      .getKernelMatrix()

    println("---------------------------------------------------------------")
    println("Calculating covariance matrix between training and test points")
    val crossKernel = SVMKernel.crossKernelMatrix(
      trainingData, test,
      covariance.evaluate)

    val aByT = kMat\crossKernel

    val hTest = DenseMatrix.vertcat(test.map(featureMap(_).asDenseMatrix):_*)
    val hTestRes = hTest - (H.t*aByT).t

    val predictiveMean = hTest*bGLS + (resGLS.t*aByT).t

    val predictiveCovariance = kernelTest - crossKernel.t*aByT + hTestRes*(hAh\hTestRes.t)

    MatrixTRV(
      npoints - num_latent_features,
      predictiveMean, predictiveCovariance,
      sigmaGLS)
  }


  /**
    * Draw three predictions from the posterior predictive distribution
    * 1) Mean or MAP estimate Y
    * 2) Y- : The lower error bar estimate (mean - sigma*stdDeviation)
    * 3) Y+ : The upper error bar. (mean + sigma*stdDeviation)
    **/
  override def predictionWithErrorBars[U <: Seq[I]](testData: U, sigma: Int) = {
    val posterior = predictiveDistribution(testData)

    val posterierMean = posterior.m
    val mean = testData.indices.toStream.map(index => posterierMean(index,::).t)

    val (lower, upper) = posterior.underlyingDist.confidenceInterval(sigma.toDouble)

    val lowerErrorBars = testData.indices.toStream.map(index => lower(index,::).t)
    val upperErrorBars = testData.indices.toStream.map(index => upper(index,::).t)


    println("Generating error bars")

    val preds = mean.zip(lowerErrorBars.zip(upperErrorBars)).map(t => (t._1, t._2._1, t._2._2))
    (testData zip preds).map(i => (i._1, i._2._1, i._2._2, i._2._3))
  }

  /**
    * Predict the value of the
    * target variable given a
    * point.
    *
    * */
  override def predict(point: I) = predictiveDistribution(Seq(point)).m.toDenseVector

  override def persist(state: Map[String, Double]) = {
    setState(state)

    val primitives = MVStudentsTModel.inferencePrimitives(covariance, noiseModel, trainingData, H, D)

    kernelMatrixCache = primitives._1
    bGLSCache = primitives._2
    sigmaGLSCache = primitives._3
    residualGLSCache = primitives._4
    hAhCache = primitives._5

    caching = true
  }

  def unpersist(): Unit = {

    kernelMatrixCache = null
    bGLSCache = null
    sigmaGLSCache = null
    residualGLSCache = null
    hAhCache = null

    caching = false
  }

  /**
    * Calculates the marginalized negative log likelihood of the data
    * in the Multivariate Students' T model.
    *
    * @param h       The value of the hyper-parameters in the configuration space
    * @param options Optional parameters about configuration
    * @return Configuration Energy E(h) = - log p(D| H, K) where
    *         H = &phi;(X) (design matrix) and K = C(X, X) (kernel matrix).
    * */
  override def energy(h: Map[String, Double], options: Map[String, String]) = {
    setState(h)
    val effectiveTrainingKernel: LocalScalarKernel[I] = covariance + noiseModel
    val A = SVMKernel.buildSVMKernelMatrix(
      trainingData, trainingData.length,
      effectiveTrainingKernel.evaluate)
      .getKernelMatrix()

    val log_likelihood_term1 = 0.5*num_outputs*log(det(A))

    lazy val h_A_h = H.t*(A\H)

    val log_likelihood_term2 = 0.5*num_outputs*log(det(h_A_h))

    lazy val d_G_d = D.t*(A\(D - (H*h_A_h*H.t)*(A\D)))

    val log_likelihood_term3 = 0.5*(npoints - num_latent_features)*log(det(d_G_d))

    log_likelihood_term1 + log_likelihood_term2 + log_likelihood_term3
  }
}

object MVStudentsTModel {

  private val logger = Logger.getLogger(this.getClass)

  type InferencePrimitives =
    (DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double])

  /**
    * Returns a set of matrices which are the starting points for
    * computing the predictive distribution of unseen data. Values
    * returned by this method can be cached for speeding up high throughput
    * inference tasks.
    *
    * @tparam I The type of the input features
    * @param covariance The covariance kernel as a [[LocalScalarKernel]]
    * @param noiseModel The covariance kernel of the measurement noise also as a [[LocalScalarKernel]]
    * @param trainingData The training data features as a sequence of [[I]]
    * @param H The design matrix resulting from applying the basis feature map &phi;(.) to the data matrix X
    * @param D The training data outputs as a [[DenseMatrix]]
    * */
  def inferencePrimitives[I: ClassTag](
    covariance: LocalScalarKernel[I], noiseModel: LocalScalarKernel[I],
    trainingData: Seq[I], H: DenseMatrix[Double], D: DenseMatrix[Double]): InferencePrimitives = {

    val (npoints, num_latent_features) = (trainingData.length, H.cols)
    val effectiveTrainingKernel: LocalScalarKernel[I] = covariance + noiseModel

    println("---------------------------------------------------------------")
    println("Calculating covariance matrix for training points")
    val A = SVMKernel.buildSVMKernelMatrix(
      trainingData, trainingData.length,
      effectiveTrainingKernel.evaluate)
      .getKernelMatrix()

    lazy val (abyH, abyD) = (A\H, A\D)

    val h_A_h = H.t*abyH
    val b_hat = h_A_h\(H.t*abyD)

    val residual_gls = D - H*b_hat

    val s_hat = residual_gls.t*(A\residual_gls)
    (A, b_hat, s_hat.mapValues(v => v/(npoints-num_latent_features).toDouble), residual_gls, h_A_h)

  }

  def apply[T, I: ClassTag](
    cov: LocalScalarKernel[I],
    noise: LocalScalarKernel[I],
    featureMap: DataPipe[I, DenseVector[Double]])(
    trainingdata: T, num: Int, num_outputs: Int)(
    implicit transform: DataPipe[T, Seq[(I, DenseVector[Double])]]) = {

    new MVStudentsTModel[T, I](cov, noise, trainingdata, num, num_outputs, featureMap) {
      /**
        * Convert from the underlying data structure to
        * Seq[(I, Y)] where I is the index set of the GP
        * and Y is the value/label type.
        **/
      override def dataAsSeq(data: T) = transform(data)
    }

  }

}