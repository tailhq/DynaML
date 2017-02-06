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
package io.github.mandar2812.dynaml.probability.distributions

import breeze.numerics._

import math.Pi
import breeze.linalg._
import breeze.stats.distributions._

import scala.runtime.ScalaRunTime

/**
  * Represents a multivariate students t distribution
  * @author mandar2812
  *
  * @param mu The degrees of freedom
  * @param mean The mean vector
  * @param covariance Covariance matrix
  */
case class MultivariateStudentsT(
  mu: Double,
  mean: DenseVector[Double],
  covariance : DenseMatrix[Double])(implicit rand: RandBasis = Rand)
  extends ContinuousDistr[DenseVector[Double]] with Moments[DenseVector[Double], DenseMatrix[Double]] {

  assert(mu > 2.0, "Parameter mu in Multivariate Students T must be greater than 2.0")

  def draw() = {
    val z: DenseVector[Double] = DenseVector.rand(mean.length, new StudentsT(mu))
    (root * z) += mean
  }

  private val root: DenseMatrix[Double] = cholesky(covariance)

  override def toString() =  ScalaRunTime._toString(this)

  override def unnormalizedLogPdf(t: DenseVector[Double]) = {
    val centered = t - mean
    val slv = covariance \ centered

    -0.5*(mu+mean.length)*log(1.0 + ((slv dot centered) / (mu - 2.0)))

  }

  override lazy val logNormalizer = {
    // determinant of the cholesky decomp is the sqrt of the determinant of the cov matrix
    // this is the log det of the cholesky decomp
    val det = sum(log(diag(root)))
    ((mean.length/2) * (log(mu - 2.0) + log(Pi))) + det + lgamma(mu/2.0) - lgamma((mu+mean.length)/2.0)
  }

  def variance = covariance*(mu/(mu-2.0))

  def mode = mean

  lazy val entropy = {
    sum(log(diag(root))) + (mean.length/2.0)*log(mu*Pi) + lbeta(mean.length/2.0, mu/2.0) - lgamma(mean.length/2.0) +
      (digamma((mu+mean.length)/2.0) - digamma(mu/2.0))*(mu+mean.length)/2.0
  }
}


