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
package io.github.mandar2812.dynaml.probability

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.{ContinuousDistr, Gaussian, MultivariateGaussian}
import io.github.mandar2812.dynaml.algebra.{PartitionedPSDMatrix, PartitionedVector}
import io.github.mandar2812.dynaml.analysis.{PartitionedVectorField, VectorField}
import io.github.mandar2812.dynaml.probability.distributions.{BlockedMultiVariateGaussian, MatrixNormal}
import spire.implicits._
import spire.algebra.Field

abstract class AbstractGaussianRV[T, V] extends ContinuousDistrRV[T]

/**
  * @author mandar2812 on 26/7/16.
  * */
case class GaussianRV(mu: Double, sigma: Double) extends AbstractGaussianRV[Double, Double] {
  override val underlyingDist = new Gaussian(mu, sigma)
}

case class MultGaussianRV(
  mu: DenseVector[Double], covariance: DenseMatrix[Double])(
  implicit ev: Field[DenseVector[Double]])
  extends AbstractGaussianRV[DenseVector[Double], DenseMatrix[Double]] {

  override val underlyingDist = MultivariateGaussian(mu, covariance)

}

object MultGaussianRV {

  def apply(num_dim: Int)(mu: DenseVector[Double], covariance: DenseMatrix[Double]) = {
    assert(
      num_dim == mu.length,
      "Number of dimensions of vector space must match the number of elements of mean")

    implicit val ev = VectorField(num_dim)

    new MultGaussianRV(mu, covariance)
  }
}

case class MultGaussianPRV(mu: PartitionedVector, covariance: PartitionedPSDMatrix)(
  implicit ev: Field[PartitionedVector])
  extends AbstractGaussianRV[PartitionedVector, PartitionedPSDMatrix] {

  override val underlyingDist: BlockedMultiVariateGaussian = BlockedMultiVariateGaussian(mu, covariance)

}

object MultGaussianPRV {

  def apply(num_dim: Long, nE: Int)(mu: PartitionedVector, covariance: PartitionedPSDMatrix) = {
    assert(
      num_dim == mu.rows,
      "Number of dimensions of vector space must match the number of elements of mean")

    implicit val ev = PartitionedVectorField(num_dim, nE)

    new MultGaussianPRV(mu, covariance)
  }

}

case class MatrixNormalRV(
  m: DenseMatrix[Double], u: DenseMatrix[Double],
  v: DenseMatrix[Double]) extends AbstractGaussianRV[
  DenseMatrix[Double], (DenseMatrix[Double], DenseMatrix[Double])] {

  override val underlyingDist = MatrixNormal(m, u, v)
}