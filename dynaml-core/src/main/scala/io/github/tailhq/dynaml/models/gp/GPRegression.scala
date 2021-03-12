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
package io.github.tailhq.dynaml.models.gp

import breeze.linalg.DenseVector
import io.github.tailhq.dynaml.algebra.PartitionedVector
import io.github.tailhq.dynaml.evaluation.RegressionMetrics
import io.github.tailhq.dynaml.kernels.{DiracKernel, LocalScalarKernel, CovarianceFunction => CovFunc}
import io.github.tailhq.dynaml.pipes.{DataPipe, StreamDataPipe}

/**
  *
  * @author tailhq
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
  cov: LocalScalarKernel[DenseVector[Double]],
  noise: LocalScalarKernel[DenseVector[Double]] = new DiracKernel(1.0),
  trainingdata: Seq[(DenseVector[Double], Double)],
  meanFunc: DataPipe[DenseVector[Double], Double] = DataPipe(_ => 0.0)) extends
AbstractGPRegressionModel[Seq[(DenseVector[Double], Double)],
  DenseVector[Double]](cov, noise, trainingdata,
  trainingdata.length, meanFunc){

  /**
    * Setting a validation set is optional in case
    * one wants to calculate joint marginal likelihood of the
    * training and validation data as the objective function for
    * hyper-parameter optimization. While retaining just the
    * training data set for final calculating [[predictiveDistribution]]
    * during final deployment.
    * */
  protected var validationSet: Seq[(DenseVector[Double], Double)] = Seq()

  /**
    * Accessor method for [[validationSet]]
    * */
  def _validationSet = validationSet

  /**
    * Set the validation data, optionally append it to the existing validation data
    *
    * @param v data
    * @param append Defaults to false
    * */
  def validationSet_(v: Seq[(DenseVector[Double], Double)], append: Boolean = false) =
    if(append) validationSet ++= v else validationSet = v


  protected lazy val validationDataFeatures = validationSet.map(_._1)

  protected lazy val validationDataLabels = PartitionedVector(
    validationSet.toStream.map(_._2),
    trainingData.length.toLong, _blockSize
  )
  /**
    * Assigning a value to the [[processTargets]] data pipe
    * can be useful in cases where we need to
    * perform operations such as de-normalizing
    * the predicted and actual targets to their original
    * scales.
    *
    * */
  @deprecated("scheduled to be removed by DynaML 2.x")
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
    * Currently this defaults to RMSE calculated
    * on the validation data.
    * */
  @deprecated("sscheduled to be removed by DynaML 2.x")
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
    * [[io.github.tailhq.dynaml.optimization.GloballyOptimizable]]
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
    case _ => super.calculateEnergyPipe(h, options)(
      trainingData ++ validationDataFeatures,
      PartitionedVector.vertcat(trainingDataLabels, validationDataLabels)
    )
  }

  /**
    * Calculates the gradient energy of the configuration and
    * subtracts this from the current value of h to yield a new
    * hyper-parameter configuration.
    *
    * Over ride this function if you aim to implement a gradient based
    * hyper-parameter optimization routine like ML-II
    *
    * @param h The value of the hyper-parameters in the configuration space
    * @return Gradient of the objective function (marginal likelihood) as a Map
    **/
  override def gradEnergy(h: Map[String, Double]) = validationSet.length match {
    case 0 => super.gradEnergy(h)
    case _ => super.calculateGradEnergyPipe(h)(
      trainingData ++ validationDataFeatures,
      PartitionedVector.vertcat(trainingDataLabels, validationDataLabels)
    )
  }
}
