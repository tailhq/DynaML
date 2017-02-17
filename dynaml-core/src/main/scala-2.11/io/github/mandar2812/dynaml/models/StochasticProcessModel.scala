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

import io.github.mandar2812.dynaml.kernels.CovarianceFunction
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.probability.ContinuousRandomVariable
import org.apache.log4j.Logger

/**
  * date 26/08/16.
  * High Level description of a stochastic process based predictive model.
  *
  * @author mandar2812
  * @tparam T The underlying data structure storing the training & test data.
  * @tparam I The type of the index set (i.e. Double for time series, DenseVector for GP regression)
  * @tparam Y The type of the output label
  * @tparam W Implementing class of the posterior distribution
  */
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
  * @author mandar2812
  *
  * Processes which can be specified by upto second order statistics i.e. mean and covariance
  * @tparam T The underlying data structure storing the training & test data.
  * @tparam I The type of the index set (i.e. Double for time series, DenseVector for GP regression)
  * @tparam Y The type of the output label
  * @tparam K The type returned by the kernel function.
  * @tparam M The data structure holding the kernel/covariance matrix
  * @tparam W Implementing class of the posterior distribution
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
  * @author mandar2812 date: 11/10/2016
  *
  * Blueprint for a continuous valued stochastic process, abstracts away the behavior
  * common to sub-classes such as [[io.github.mandar2812.dynaml.models.gp.GPRegression]],
  * [[io.github.mandar2812.dynaml.models.stp.StudentTRegression]] and others.
  *
  */
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