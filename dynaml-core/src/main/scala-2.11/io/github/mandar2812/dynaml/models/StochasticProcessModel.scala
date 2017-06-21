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
package io.github.mandar2812.dynaml.models

import breeze.linalg.DenseVector
import breeze.stats.distributions.{ContinuousDistr, Moments}
import io.github.mandar2812.dynaml.kernels.CovarianceFunction
import io.github.mandar2812.dynaml.models.gp.{AbstractGPRegressionModel, GaussianProcessMixture}
import io.github.mandar2812.dynaml.models.stp.{AbstractSTPRegressionModel, MVStudentsTModel, MVTMixture, StudentTProcessMixture}
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.probability._
import io.github.mandar2812.dynaml.probability.distributions.HasErrorBars
import org.apache.log4j.Logger
import spire.algebra.{InnerProductSpace, VectorSpace}

import scala.reflect.ClassTag

/**
  * High Level description of a stochastic process based predictive model.
  *
  * @tparam T The underlying data structure storing the training & test data.
  * @tparam I The type of the index set (i.e. Double for time series, DenseVector for GP regression)
  * @tparam Y The type of the output label
  * @tparam W Implementing class of the posterior distribution
  * @author mandar2812 date 26/08/16.
  *
  * */
trait StochasticProcessModel[T, I, Y, W] extends Model[T, I, Y] {

  /** Calculates posterior predictive distribution for
    * a particular set of test data points.
    *
    * @param test A Sequence or Sequence like data structure
    *             storing the values of the input patters.
    * */
  def predictiveDistribution[U <: Seq[I]](test: U): W


  /**
    * Convert from the underlying data structure to
    * Seq[(I, Y)] where I is the index set of the GP
    * and Y is the value/label type.
    * */

  def dataAsSeq(data: T): Seq[(I,Y)]

  /**
    * Convert from the underlying data structure to
    * Seq[I] where I is the index set of the GP
    * */
  def dataAsIndexSeq(data: T): Seq[I] = dataAsSeq(data).map(_._1)


}

/**
  * Processes which can be specified by upto second order statistics i.e. mean and covariance
  * @tparam T The underlying data structure storing the training & test data.
  * @tparam I The type of the index set (i.e. Double for time series, DenseVector for GP regression)
  * @tparam Y The type of the output label
  * @tparam K The type returned by the kernel function.
  * @tparam M The data structure holding the kernel/covariance matrix
  * @tparam W Implementing class of the posterior distribution
  * @author mandar2812
  *
  * */
trait SecondOrderProcessModel[T, I, Y, K, M, W] extends StochasticProcessModel[T, I, Y, W] {

  /**
    * Mean Function: Takes a member of the index set (input)
    * and returns the corresponding mean of the distribution
    * corresponding to input.
    * */
  val mean: DataPipe[I, Y]

  /**
    * Underlying covariance function of the
    * Gaussian Processes.
    * */
  val covariance: CovarianceFunction[I, K, M]


}

/**
  * Blueprint for a continuous valued stochastic process, abstracts away the behavior
  * common to sub-classes such as [[io.github.mandar2812.dynaml.models.gp.GPRegression]],
  * [[io.github.mandar2812.dynaml.models.stp.StudentTRegression]] and others.
  *
  * @author mandar2812 date: 11/10/2016
  *
  * */
abstract class ContinuousProcessModel[T, I, Y, W <: ContinuousRandomVariable[_]]
  extends StochasticProcessModel[T, I, Y, W] {

  private val logger = Logger.getLogger(this.getClass)

  /**
    * Represents how many intervals away from the mean the error bars lie, defaults to 1
    */
  private var errorSigma: Int = 1

  def _errorSigma = errorSigma

  def errorSigma_(s: Int) = errorSigma = s

  /**
    * Draw three predictions from the posterior predictive distribution
    * 1) Mean or MAP estimate Y
    * 2) Y- : The lower error bar estimate (mean - sigma*stdDeviation)
    * 3) Y+ : The upper error bar. (mean + sigma*stdDeviation)
    **/
  def predictionWithErrorBars[U <: Seq[I]](testData: U, sigma: Int): Seq[(I, Y, Y, Y)]

  /**
    * Returns a prediction with error bars for a test set of indexes and labels.
    * (Index, Actual Value, Prediction, Lower Bar, Higher Bar)
    * */
  def test(testData: T): Seq[(I, Y, Y, Y, Y)] = {
    logger.info("Generating predictions for test set")
    //Calculate the posterior predictive distribution for the test points.
    val predictionWithError = predictionWithErrorBars(dataAsIndexSeq(testData), errorSigma)
    //Collate the test data with the predictions and error bars
    dataAsSeq(testData)
      .zip(predictionWithError)
      .map(i => (
        i._1._1, i._1._2,
        i._2._2, i._2._3,
        i._2._4))
  }

}

abstract class StochasticProcessMixtureModel[
I, Y, W <: ContinuousMixtureRV[_, _]] extends
  ContinuousProcessModel[Seq[(I, Y)], I, Y, W]


object StochasticProcessMixtureModel {

  def apply[T, I: ClassTag](
    component_processes: Seq[AbstractGPRegressionModel[T, I]],
    weights: DenseVector[Double]) =
    new GaussianProcessMixture[T, I](component_processes, weights)

  def apply[T, I: ClassTag](
    component_processes: Seq[AbstractSTPRegressionModel[T, I]],
    weights: DenseVector[Double]) =
    new StudentTProcessMixture[T, I](component_processes, weights)


  def apply[T, I: ClassTag](
    component_processes: Seq[MVStudentsTModel[T, I]],
    weights: DenseVector[Double]) =
    new MVTMixture[T, I](component_processes, weights)

}

/**
  * A process which is a multinomial mixture of
  * continuous component processes.
  * @tparam I The type of the index set (i.e. Double for time series, DenseVector for GP regression)
  * @tparam Y The type of the output label
  * @tparam W1 Implementing class of the posterior distribution for the base processes
  *           should inherit from [[ContinuousMixtureRV]]
  * @author mandar2812 date 19/06/2017
  * */
abstract class ContinuousMixtureModel[
T, I: ClassTag, Y, YDomain,
W1 <: ContinuousDistrRV[YDomain],
BaseProcesses <: ContinuousProcessModel[T, I, Y, W1]](
  val component_processes: Seq[BaseProcesses],
  val weights: DenseVector[Double]) extends
  StochasticProcessMixtureModel[I, Y, ContinuousDistrMixture[YDomain, W1]] {

  /** Calculates posterior predictive distribution for
    * a particular set of test data points.
    *
    * @param test A Sequence or Sequence like data structure
    *             storing the values of the input patters.
    * */
  override def predictiveDistribution[U <: Seq[I]](test: U) =
    ContinuousDistrMixture[YDomain, W1](component_processes.map(_.predictiveDistribution(test)), weights)
}

/**
  * A multinomial mixture of component processes, each
  * of which can output predictive distributions which have
  * error bars around the mean/mode.
  *
  * @tparam T The training data type of each component
  * @tparam I The input feature type accepted by each component
  * @tparam Y The type of the output label
  * @tparam YDomain The type of a collection of outputs, e.g. vector
  * @tparam YDomainVar The type of the second moment (variance) returned
  *                    by the predictive distribution of each component process
  * @tparam BaseDistr The type of the predictive distribution of each process.
  * @tparam W1 The random variable type returned by the [[predictiveDistribution()]] method
  *            of each component.
  * @tparam BaseProcesses The type of the stochastic process components
  *
  *
  * @param component_processes The stochastic processes which form the components of the mixture
  * @param weights The probability weights assigned to each component.
  * @author mandar2812 date 19/06/2017
  * */
abstract class GenContinuousMixtureModel[
T, I: ClassTag, Y, YDomain, YDomainVar,
BaseDistr <: ContinuousDistr[YDomain] with Moments[YDomain, YDomainVar] with HasErrorBars[YDomain],
W1 <: ContinuousRVWithDistr[YDomain, BaseDistr],
BaseProcesses <: ContinuousProcessModel[T, I, Y, W1]](
  val component_processes: Seq[BaseProcesses],
  val weights: DenseVector[Double]) extends
  StochasticProcessMixtureModel[I, Y, ContMixtureRVBars[YDomain, YDomainVar, BaseDistr]] {

  private val logger = Logger.getLogger(this.getClass)

  /**
    * The training data
    * */
  override protected val g: Seq[(I, Y)] = Seq()

  /**
    * Convert from the underlying data structure to
    * Seq[(I, Y)] where I is the index set of the GP
    * and Y is the value/label type.
    * */
  override def dataAsSeq(data: Seq[(I, Y)]) = data

  /**
    * Predict the value of the
    * target variable given a
    * point.
    *
    * */
  override def predict(point: I) = predictionWithErrorBars(Seq(point), 1).head._2

  protected def toStream(y: YDomain): Stream[Y]

  protected def getVectorSpace(num_dim: Int): VectorSpace[YDomain, Double]


  /** Calculates posterior predictive distribution for
    * a particular set of test data points.
    *
    * @param test A Sequence or Sequence like data structure
    *             storing the values of the input patters.
    * */
  override def predictiveDistribution[U <: Seq[I]](test: U): ContMixtureRVBars[YDomain, YDomainVar, BaseDistr] =
    ContinuousDistrMixture[YDomain, YDomainVar, BaseDistr](
      component_processes.map(_.predictiveDistribution(test).underlyingDist),
      weights)(getVectorSpace(test.length))

  /**
    * Draw three predictions from the posterior predictive distribution
    * 1) Mean or MAP estimate Y
    * 2) Y- : The lower error bar estimate (mean - sigma*stdDeviation)
    * 3) Y+ : The upper error bar. (mean + sigma*stdDeviation)
    * */
  override def predictionWithErrorBars[U <: Seq[I]](testData: U, sigma: Int) = {

    val posterior = predictiveDistribution(testData)

    val mean = toStream(posterior.underlyingDist.mean)

    val (lower, upper) = posterior.underlyingDist.confidenceInterval(sigma.toDouble)

    val lowerErrorBars = toStream(lower)
    val upperErrorBars = toStream(upper)

    logger.info("Generating error bars")

    val preds = mean.zip(lowerErrorBars.zip(upperErrorBars)).map(t => (t._1, t._2._1, t._2._2))
    (testData zip preds).map(i => (i._1, i._2._1, i._2._2, i._2._3))

  }
}