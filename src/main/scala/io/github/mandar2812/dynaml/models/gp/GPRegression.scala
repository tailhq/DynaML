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

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.kernels.{DiracKernel, CovarianceFunction => CovFunc}
import io.github.mandar2812.dynaml.pipes.{DataPipe, StreamDataPipe}

/**
  *
  * @author mandar2812
  * date: 17/11/15.
  *
  * Class representing Gaussian Process regression models
  *
  * y = f(x) + e
  * f(x) ~ GP(0, cov(X,X))
  * e|f(x) ~ N(f(x), noise(X,X))
  *
  * Constructor Parameters:
  *
  * @param cov The covariance/kernel function
  *            as an appropriate subtype of [[CovFunc]]
  *
  * @param noise The kernel function describing the
  *              noise model, defaults to [[DiracKernel]].
  *
  * @param trainingdata The data structure containing the
  *                     training data i.e. [[Seq]] of [[Tuple2]]
  *                     of the form (features, target)
  *
  */
class GPRegression(
  cov: CovFunc[DenseVector[Double], Double, DenseMatrix[Double]],
  noise: CovFunc[DenseVector[Double], Double, DenseMatrix[Double]] = new DiracKernel(1.0),
  trainingdata: Seq[(DenseVector[Double], Double)]) extends
AbstractGPRegressionModel[Seq[(DenseVector[Double], Double)],
  DenseVector[Double]](cov, noise, trainingdata,
  trainingdata.length){

  /**
    * Setting a validation set is optional in case
    * one wants to use some function of metrics like correltaion or rmse
    * as hyper-parameter optimization objective functions.
    * */
  var validationSet: Seq[(DenseVector[Double], Double)] = Seq()

  /**
    * Setting a data pipe to process predicted and
    * actual target values can be useful in cases where
    * one needs to perform operations such as de-normalizing
    * the predicted and actual targets to their original
    * scales.
    *
    * */
  var processTargets: DataPipe[
    Stream[(Double, Double)],
    Stream[(Double, Double)]] =
    StreamDataPipe((predictionCouple: (Double, Double)) =>
      identity(predictionCouple))

  /**
    * If one uses a non empty validation set, then
    * the user can set a custom function of
    * the validation predictions and targets as
    * the objective function for the hyper-parameter
    * optimization routine.
    *
    * Currently this defaults to RMSE*(1-CC) calculated
    * on the validation data.
    * */
  var scoresToEnergy: DataPipe[Stream[(Double, Double)], Double] =
    DataPipe((scoresAndLabels) => {

      val metrics = new RegressionMetrics(
        scoresAndLabels.toList,
        scoresAndLabels.length
      )

      metrics.rmse
    })

  /**
    * Convert from the underlying data structure to
    * Seq[(I, Y)] where I is the index set of the GP
    * and Y is the value/label type.
    **/
  override def dataAsSeq(data: Seq[(DenseVector[Double], Double)]) = data

  /**
    * Calculates the energy of the configuration, required
    * for global optimization routines.
    *
    * Defaults to the base implementation in
    * [[io.github.mandar2812.dynaml.optimization.GloballyOptimizable]]
    * in case a validation set is not specified
    * through the [[validationSet]] variable.
    *
    * @param h The value of the hyper-parameters in the configuration space
    * @param options Optional parameters about configuration
    * @return Configuration Energy E(h)
    * */
  override def energy(h: Map[String, Double],
                      options: Map[String, String]): Double = validationSet.length match {
    case 0 => super.energy(h, options)
    case _ =>
      // Calculate regression metrics on validation set
      // Return some function of kpi as energy

      val resultsToScores = DataPipe(
        (res: Seq[(DenseVector[Double], Double, Double, Double, Double)]) =>
          res.map(i => (i._3, i._2)).toStream)

      (resultsToScores >
        processTargets >
        scoresToEnergy) run
        this.test(validationSet)
  }

}
