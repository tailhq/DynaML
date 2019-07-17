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
package io.github.mandar2812.dynaml.modelpipe

import breeze.linalg._
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.kernels.LocalScalarKernel
import io.github.mandar2812.dynaml.probability.MultGaussianRV
import io.github.mandar2812.dynaml.models.gp.{AbstractGPRegressionModel, GPBasisFuncRegressionModel}
import io.github.mandar2812.dynaml.pipes.{DataPipe, DataPipe2}

import scala.reflect.ClassTag

/**
  * <h3>GP Pipes</h3>
  *
  * A data pipe which can spawn a Gaussian Process regression model
  * from a provided training data set.
  *
  * @tparam IndexSet Type of features of each data pattern
  * @tparam Source Input data type
  * @param pre A function which converts the input data
  *            into a scala [[Seq]] of [[IndexSet]]
  *            and [[Double]] pairs.
  * @param covariance The covariance function of the resultant GP,
  *            as an instance of [[LocalScalarKernel]] defined on
  *            the [[IndexSet]] type.
  * @param noise The measurement noise of the output signal/data, also
  *          as in instance of [[LocalScalarKernel]]
  * @param order Size of the auto-regressive time lag of the output signal
  *              that is used to create the training data. Ignore if not working
  *              with GP-NAR or GP-NARX models.
  * @param ex Size of the auto-regressive time lag of the exogenous inputs
  *           that is used to create the training data. Ignore if not working
  *           with GP-NARX models.
  * @param meanFunc The trend function of the resulting GP model, as an instance
  *                 of [[DataPipe]].
  * @author mandar2812 on 15/6/16.
  * */
class GPRegressionPipe[Source, IndexSet: ClassTag](
  pre: DataPipe[Source, Seq[(IndexSet, Double)]],
  val covariance: LocalScalarKernel[IndexSet],
  val noise: LocalScalarKernel[IndexSet],
  meanFunc: DataPipe[IndexSet, Double] = DataPipe((_: IndexSet) => 0.0))
  extends ModelPipe[
    Source, Seq[(IndexSet, Double)], IndexSet, Double,
    AbstractGPRegressionModel[Seq[(IndexSet, Double)], IndexSet]] {

  override val preProcess: DataPipe[Source, Seq[(IndexSet, Double)]] = pre

  implicit val transform = identityPipe[Seq[(IndexSet, Double)]]

  override def run(data: Source): AbstractGPRegressionModel[Seq[(IndexSet, Double)], IndexSet] =
    AbstractGPRegressionModel(covariance, noise, meanFunc)(preProcess(data), 0)

}

/**
  * <h3>GP Basis Function Pipe</h3>
  *
  * A data pipe which can spawn a Gaussian Process Basis Function
  * regression model from a provided training data set.
  *
  * @tparam IndexSet Type of features of each data pattern
  * @tparam Source Input data type
  * @param pre A function which converts the input data
  *            into a scala [[Seq]] of [[IndexSet]]
  *            and [[Double]] pairs.
  * @param covariance The covariance function of the resultant GP,
  *            as an instance of [[LocalScalarKernel]] defined on
  *            the [[IndexSet]] type.
  * @param noise The measurement noise of the output signal/data, also
  *          as in instance of [[LocalScalarKernel]]
  * @param order Size of the auto-regressive time lag of the output signal
  *              that is used to create the training data. Ignore if not working
  *              with GP-NAR or GP-NARX models.
  * @param ex Size of the auto-regressive time lag of the exogenous inputs
  *           that is used to create the training data. Ignore if not working
  *           with GP-NARX models.
  * 
  * @param basisFunc A basis function representation for the input features,
  *                  represented as a [[DataPipe]].
  *
  * @param basis_param_prior A Gaussian prior on the basis function trend coefficients.
  *
  * @author mandar2812 date 2017/08/09
  * */
class GPBasisFuncRegressionPipe[Source, IndexSet: ClassTag](
  pre: DataPipe[Source, Seq[(IndexSet, Double)]],
  val covariance: LocalScalarKernel[IndexSet],
  val noise: LocalScalarKernel[IndexSet],
  basisFunc: DataPipe[IndexSet, DenseVector[Double]],
  basis_param_prior: MultGaussianRV)
  extends ModelPipe[
    Source, Seq[(IndexSet, Double)], IndexSet, Double,
    GPBasisFuncRegressionModel[Seq[(IndexSet, Double)], IndexSet]] {

  override val preProcess: DataPipe[Source, Seq[(IndexSet, Double)]] = pre

  implicit val transform = identityPipe[Seq[(IndexSet, Double)]]

  override def run(data: Source): GPBasisFuncRegressionModel[Seq[(IndexSet, Double)], IndexSet] =
    AbstractGPRegressionModel(covariance, noise, basisFunc, basis_param_prior)(preProcess(data), 0)

}


object GPRegressionPipe {

  /**
    * Convenience method for creating [[GPRegressionPipe]] instances
    * */
  def apply[Source, IndexSet: ClassTag](
    pre: DataPipe[Source, Seq[(IndexSet, Double)]],
    covariance: LocalScalarKernel[IndexSet], noise: LocalScalarKernel[IndexSet],
    meanFunc: DataPipe[IndexSet, Double] = DataPipe((_: IndexSet) => 0.0)) =
    new GPRegressionPipe[Source, IndexSet](pre, covariance, noise, meanFunc)
}


/**
  * <h3>GP Pipes: Alternate</h3>
  * A [[DataPipe2]] which takes a data set,
  * a trend and outputs a GP Regression model.
  *
  * @tparam IndexSet Type of features of each data pattern
  * @param covariance The covariance function of the resultant GP,
  *            as an instance of [[LocalScalarKernel]] defined on
  *            the [[IndexSet]] type.
  * @param noise The measurement noise of the output signal/data, also
  *          as in instance of [[LocalScalarKernel]]
  *
  * */
class GPRegressionPipe2[IndexSet: ClassTag](
  val covariance: LocalScalarKernel[IndexSet],
  val noise: LocalScalarKernel[IndexSet]) extends DataPipe2[
  Seq[(IndexSet, Double)], DataPipe[IndexSet, Double],
  AbstractGPRegressionModel[Seq[(IndexSet, Double)], IndexSet]] {

  implicit val transform = identityPipe[Seq[(IndexSet, Double)]]

  override def run(data: Seq[(IndexSet, Double)], trend: DataPipe[IndexSet, Double]) =
    AbstractGPRegressionModel(covariance, noise, trend)(data, data.length)
}


object GPRegressionPipe2 {

  /**
    * Convenience method for creating [[GPRegressionPipe2]] instances
    * */
  def apply[IndexSet: ClassTag](
    covariance: LocalScalarKernel[IndexSet],
    noise: LocalScalarKernel[IndexSet]): GPRegressionPipe2[IndexSet] =
    new GPRegressionPipe2(covariance, noise)
}
