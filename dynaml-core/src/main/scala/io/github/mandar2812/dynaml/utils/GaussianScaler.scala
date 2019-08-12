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
package io.github.mandar2812.dynaml.utils

import breeze.linalg.{DenseMatrix, DenseVector, cholesky, inv}
import io.github.mandar2812.dynaml.pipes.{ReversibleScaler, Scaler}

/**
  * Scales attributes of a vector pattern using the sample mean and variance of
  * each dimension. This assumes that there is no covariance between the data
  * dimensions.
  *
  * @param mean Sample mean of the data
  * @param sigma Sample variance of each data dimension
  * @author mandar2812 date: 17/6/16.
  *
  * */
case class GaussianScaler(mean: DenseVector[Double], sigma: DenseVector[Double])
  extends ReversibleScaler[DenseVector[Double]]{
  override val i: Scaler[DenseVector[Double]] =
    Scaler((pattern: DenseVector[Double]) => (pattern *:* sigma) + mean)

  override def run(data: DenseVector[Double]): DenseVector[Double] = (data-mean) /:/ sigma

  def apply(r: Range): GaussianScaler = GaussianScaler(mean(r), sigma(r))

  def apply(index: Int): UnivariateGaussianScaler = UnivariateGaussianScaler(mean(index), sigma(index))

  def ++(other: GaussianScaler) =
    GaussianScaler(
      DenseVector(this.mean.toArray++other.mean.toArray),
      DenseVector(this.sigma.toArray++other.sigma.toArray))
}


/**
  * Scales the attributes of a data pattern using the sample mean and covariance matrix
  * calculated on the data set. This allows standardization of multivariate data sets
  * where the covariance of individual data dimensions is not negligible.
  *
  * @param mean Sample mean of data
  * @param sigma Sample covariance matrix of data.
  * */
case class MVGaussianScaler(mean: DenseVector[Double], sigma: DenseMatrix[Double])
  extends ReversibleScaler[DenseVector[Double]] {

  val sigmaInverse = cholesky(inv(sigma))

  override val i: Scaler[DenseVector[Double]] =
    Scaler((pattern: DenseVector[Double]) => (inv(sigmaInverse.t) * pattern) + mean)

  override def run(data: DenseVector[Double]): DenseVector[Double] = sigmaInverse.t * (data - mean)

  def apply(r: Range): MVGaussianScaler = MVGaussianScaler(mean(r), sigma(r,r))
}

case class UnivariateGaussianScaler(mean: Double, sigma: Double) extends ReversibleScaler[Double] {
  require(sigma > 0.0, "Std. Deviation for gaussian scaling must be strictly positive!")
  /**
    * The inverse operation of this scaling.
    *
    **/
  override val i = Scaler((pattern: Double) => (pattern*sigma) + mean)

  override def run(data: Double) = (data-mean)/sigma
}