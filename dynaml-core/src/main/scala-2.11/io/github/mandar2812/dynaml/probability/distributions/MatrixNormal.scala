package io.github.mandar2812.dynaml.probability.distributions

import breeze.linalg.{DenseMatrix, cholesky, diag, sum, trace}
import breeze.numerics.log
import breeze.stats.distributions.{ContinuousDistr, Moments, Rand, RandBasis}

import scala.math.log1p

/**
  * @author mandar date: 05/02/2017.
  * Matrix normal distribution over n &times p matrices
  *
  * @param m The mode, mean and center of the distribution
  * @param u The n &times; n covariance matrix of the rows
  * @param v The p &times; p covariance matrix of the columns
  */
case class MatrixNormal(
  m: DenseMatrix[Double],
  u: DenseMatrix[Double],
  v: DenseMatrix[Double])(
  implicit rand: RandBasis = Rand) extends
  ContinuousDistr[DenseMatrix[Double]] with
  Moments[DenseMatrix[Double], (DenseMatrix[Double], DenseMatrix[Double])]{

  private lazy val (rootu, rootv) = (cholesky(u), cholesky(v))

  private val (n,p) = (u.rows, v.cols)

  override def unnormalizedLogPdf(x: DenseMatrix[Double]) = {
    val d = x - m
    val y = rootu.t \ (rootu \ d)

    -0.5*trace(rootv.t\(rootv\(d.t*y)))
  }

  override lazy val logNormalizer = {
    val detU = sum(log(diag(rootu)))
    val detV = sum(log(diag(rootv)))

    0.5*(log(2.0*math.Pi)*n*p + detU*p + detV*n)
  }

  override def mean = m

  override def variance = (u,v)

  override def mode = m

  override def draw() = {
    val z: DenseMatrix[Double] = DenseMatrix.rand(m.rows, m.cols, rand.gaussian(0.0, 1.0))
    mean + (rootu*z*rootv.t)
  }

  lazy val entropy = {
    m.rows * m.cols * (log1p(2 * math.Pi) + sum(log(diag(rootu))) + sum(log(diag(rootv))))
  }

}
