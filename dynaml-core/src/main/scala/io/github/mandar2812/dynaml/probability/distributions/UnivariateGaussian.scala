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
package io.github.mandar2812.dynaml.probability.distributions

import breeze.linalg.{DenseMatrix, DenseVector, cholesky, diag, sum, trace}
import breeze.numerics.log
import breeze.stats.distributions._

import scala.math.log1p

/**
  * @author mandar date 20/06/2017.
  * */
class UnivariateGaussian(mu: Double, sigma: Double)
  extends Gaussian(mu, sigma) with HasErrorBars[Double] {
  override def confidenceInterval(s: Double): (Double, Double) = {
    (mu-s*sigma, mu+s*sigma)
  }
}

object UnivariateGaussian {
  def apply(mu: Double, sigma: Double) = new UnivariateGaussian(mu, sigma)

}


class MVGaussian(
  mu: DenseVector[Double],
  covariance: DenseMatrix[Double])(
  implicit rand: RandBasis = Rand) extends
  AbstractContinuousDistr[DenseVector[Double]] with
  Moments[DenseVector[Double], DenseMatrix[Double]] with
  HasErrorBars[DenseVector[Double]] {

  protected lazy val root: DenseMatrix[Double] = cholesky(covariance)

  override def mean: DenseVector[Double] = mu

  override def variance: DenseMatrix[Double] = covariance

  override def entropy: Double = {
    mean.length * log1p(2 * math.Pi) + sum(log(diag(root)))
  }

  override def mode: DenseVector[Double] = mean

  override def unnormalizedLogPdf(x: DenseVector[Double]): Double = {
    val centered = x - mean
    val slv = covariance \ centered

    -(slv dot centered) / 2.0

  }

  override def logNormalizer: Double = {
    // determinant of the cholesky decomp is the sqrt of the determinant of the cov matrix
    // this is the log det of the cholesky decomp
    val det = trace(log(root))
    mean.length/2.0 *  log(2 * math.Pi) + 0.5*det
  }

  override def draw(): DenseVector[Double] = {
    val z: DenseVector[Double] = DenseVector.rand(mean.length, rand.gaussian(0, 1))
    root * z += mean
  }

  override def confidenceInterval(s: Double): (DenseVector[Double], DenseVector[Double]) = {

    val signFlag = if(s < 0) -1.0 else 1.0

    val ones = DenseVector.ones[Double](mean.length)
    val multiplier = signFlag*s

    val bar: DenseVector[Double] = root*(ones*multiplier)

    (mean - bar, mean + bar)

  }
}

object MVGaussian {

  def apply(
    mu: DenseVector[Double],
    covariance: DenseMatrix[Double])(
    implicit rand: RandBasis = Rand): MVGaussian = new MVGaussian(mu,covariance)
}