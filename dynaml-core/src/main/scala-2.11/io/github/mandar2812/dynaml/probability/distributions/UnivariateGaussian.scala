package io.github.mandar2812.dynaml.probability.distributions

import breeze.linalg.{DenseMatrix, DenseVector, cholesky, diag, sum}
import breeze.numerics.log
import breeze.stats.distributions._

import scala.math.log1p
import scala.runtime.ScalaRunTime

/**
  * Created by mandar on 20/06/2017.
  */
class UnivariateGaussian(mu: Double, sigma: Double)
  extends Gaussian(mu, sigma) with HasErrorBars[Double] {
  override def confidenceInterval(s: Double) = {
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

  override def mean = mu

  override def variance = covariance

  override def entropy = {
    mean.length * log1p(2 * math.Pi) + sum(log(diag(root)))
  }

  override def mode = mean

  override def unnormalizedLogPdf(x: DenseVector[Double]) = {
    val centered = x - mean
    val slv = covariance \ centered

    -(slv dot centered) / 2.0

  }

  override def logNormalizer = {
    // determinant of the cholesky decomp is the sqrt of the determinant of the cov matrix
    // this is the log det of the cholesky decomp
    val det = sum(log(diag(root)))
    mean.length/2.0 *  log(2 * math.Pi) + det
  }

  override def draw() = {
    val z: DenseVector[Double] = DenseVector.rand(mean.length, rand.gaussian(0, 1))
    root * z += mean
  }

  override def confidenceInterval(s: Double) = {

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