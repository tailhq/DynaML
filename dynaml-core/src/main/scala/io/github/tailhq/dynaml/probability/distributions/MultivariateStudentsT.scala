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
package io.github.tailhq.dynaml.probability.distributions

import breeze.numerics._

import math.Pi
import breeze.linalg._
import breeze.stats.distributions._
import org.apache.spark.annotation.Experimental

import scala.runtime.ScalaRunTime

/**
  * @author mandar date 20/06/2017.
  * */
case class UnivariateStudentsT(
  mu:Double, mean: Double, sigma: Double)(
  implicit rand: RandBasis = Rand) extends
  AbstractContinuousDistr[Double] with
  Moments[Double, Double] with
  HasErrorBars[Double] {

  assert(mu > 2d, "Parameter mu in Multivariate Students T must be greater than 2")

  private val chisq = new ChiSquared(mu)

  override def variance = sigma*sigma*mu/(mu-2d)

  override def entropy =
    0.5*(mu+1d)*(digamma(0.5*(mu+1d)) - digamma(0.5*mu)) +
      log(sqrt(mu)) + lbeta(0.5*mu, 0.5)

  override def mode = mean

  override def draw() = {
    val w = math.sqrt(mu/chisq.draw())
    val z = rand.gaussian(0.0, 1.0).draw()*w
    (sigma * z) + mean
  }

  override def unnormalizedLogPdf(x: Double) = {
    val x_std = (x-mean)/sigma
    -0.5*(mu+1d)*log(1d + math.pow(x_std, 2d)/mu)
  }

  override def logNormalizer =
    lgamma(0.5*(mu+1d)) - 0.5*log(math.Pi*mu) - log(sigma) -lgamma(0.5*mu)

  override def confidenceInterval(s: Double) = {
    (mu-s*sigma, mu+s*sigma)
  }
}


/**
  * Represents a multivariate students t distribution
  * @author tailhq
  *
  * @param mu The degrees of freedom
  * @param mean The mean vector
  * @param covariance Covariance matrix
  * */
case class MultivariateStudentsT(
  mu: Double,
  mean: DenseVector[Double],
  covariance : DenseMatrix[Double])(implicit rand: RandBasis = Rand) extends
  AbstractContinuousDistr[DenseVector[Double]] with
  Moments[DenseVector[Double], DenseMatrix[Double]] with
  HasErrorBars[DenseVector[Double]] {

  assert(mu > 2.0, "Parameter mu in Multivariate Students T must be greater than 2.0")

  private val chisq = new ChiSquared(mu)

  def draw() = {
    val w = math.sqrt(mu/chisq.draw())
    val z: DenseVector[Double] = DenseVector.rand(mean.length, rand.gaussian(0.0, 1.0))*w
    (root * z) += mean
  }

  private val root: DenseMatrix[Double] = cholesky(covariance)

  override def toString() =  ScalaRunTime._toString(this)

  override def unnormalizedLogPdf(t: DenseVector[Double]) = {
    val centered = t - mean
    val slv = covariance \ centered

    -0.5*(mu+mean.length)*log(1.0 + ((slv dot centered) / mu))

  }

  override lazy val logNormalizer = {
    // determinant of the cholesky decomp is the sqrt of the determinant of the cov matrix
    // this is the log det of the cholesky decomp
    val det = sum(log(diag(root)))
    ((mean.length/2) * (log(mu) + log(Pi))) + 0.5*det + lgamma(mu/2.0) - lgamma((mu+mean.length)/2.0)
  }

  def variance = covariance*(mu/(mu-2.0))

  def mode = mean

  //TODO: Check and correct calculation of entropy for Mult Students T
  @Experimental
  lazy val entropy = {
    sum(log(diag(root))) + (mean.length/2.0)*log(mu*Pi) + lbeta(mean.length/2.0, mu/2.0) - lgamma(mean.length/2.0) +
      (digamma((mu+mean.length)/2.0) - digamma(mu/2.0))*(mu+mean.length)/2.0
  }

  override def confidenceInterval(s: Double) = {
    val signFlag = if(s < 0) -1.0 else 1.0

    val ones = DenseVector.ones[Double](mean.length)
    val multiplier = signFlag*s

    val bar: DenseVector[Double] = root*(ones*(multiplier*math.sqrt(mu/(mu-2.0))))

    (mean - bar, mean + bar)
  }
}


