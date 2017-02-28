package io.github.mandar2812.dynaml.probability.distributions


import breeze.linalg.{DenseMatrix, DenseVector, cholesky}
import breeze.numerics.sqrt
import breeze.stats.distributions._
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.utils._
import io.github.mandar2812.dynaml.algebra._
import spire.implicits._

/**
  * @author mandar2812 date: 02/01/2017.
  *
  * Univariate skew gaussian distribution
  */
case class SkewGaussian(
  alpha: Double, mu: Double = 0.0,
  sigma: Double = 1.0) extends
  SkewSymmDistribution[Double](
    Gaussian(mu, sigma), Gaussian(0.0, 1.0)) {

  override protected val w = DataPipe((x: Double) => alpha*(x-mu)/sqrt(sigma))

  override protected val warped_cutoff: Double = 0.0
}

/**
  * The univariate extended skew gaussian distribution
  * */
case class ExtendedSkewGaussian(
  alpha0: Double, alpha: Double,
  mu: Double = 0.0, sigma: Double = 1.0)(
  implicit rand: RandBasis = Rand)
  extends SkewSymmDistribution[Double](
    Gaussian(mu, sigma), Gaussian(0.0, 1.0),
    alpha0) {

  override protected val w = DataPipe((x: Double) => alpha*(x-mu)/sqrt(sigma))

  override protected val warped_cutoff: Double = alpha0*sqrt(1 + alpha*alpha)

}

/**
  * Multivariate Skew-Normal distribution as
  * specified in Azzalani et. al.
  *
  * @param alpha A breeze [[DenseVector]] which represents the skewness parameters
  *
  * @param mu The center of the distribution
  *
  * @param sigma The covariance matrix of the base multivariate gaussian.
  *
  * */
case class MultivariateSkewNormal(
  alpha: DenseVector[Double],
  mu: DenseVector[Double],
  sigma: DenseMatrix[Double]) extends
  SkewSymmDistribution[DenseVector[Double]](
    basisDistr = MultivariateGaussian(mu, sigma),
    warpingDistr = Gaussian(0.0, 1.0))(
    VectorField(alpha.length)) {

  private lazy val omega = diagonal(sigma)

  private lazy val sqrtSigma = sqrt(omega)

  override protected val w = DataPipe((x: DenseVector[Double]) => crossQuadraticForm(alpha, cholesky(sqrtSigma), x-mu))

  override protected val warped_cutoff: Double = 0.0
}

/**
  * Extended Multivariate Skew-Gaussian distribution
  * as specified in Azzalani et.al.
  *
  * @param tau Determines the cutoff of the warping function
  *
  * @param alpha A breeze [[DenseVector]] which represents the skewness parameters
  *
  * @param mu The center of the distribution
  *
  * @param sigma The covariance matrix of the base multivariate gaussian.
  * */
case class ExtendedMultivariateSkewNormal(
  tau: Double, alpha: DenseVector[Double],
  mu: DenseVector[Double], sigma: DenseMatrix[Double]) extends
  SkewSymmDistribution[DenseVector[Double]](
    basisDistr = MultivariateGaussian(mu, sigma),
    warpingDistr = Gaussian(0.0, 1.0),
    cutoff = tau)(
    VectorField(alpha.length)) {

  private lazy val omega = diagonal(sigma)

  private lazy val sqrtSigma = sqrt(omega)

  private lazy val correlationMatrix: DenseMatrix[Double] = omega\sigma

  override protected val w = DataPipe((x: DenseVector[Double]) => crossQuadraticForm(alpha, cholesky(sqrtSigma), x-mu))

  override protected val warped_cutoff: Double = tau*sqrt(1.0 + quadraticForm(cholesky(correlationMatrix), alpha))
}


/**
  * Extended Multivariate Skew-Gaussian distribution
  * as specified in Adcock and Schutes.
  *
  * @param tau Determines the cutoff of the warping function
  *
  * @param alpha A breeze [[DenseVector]] which represents the skewness parameters
  *
  * @param mu The center of the distribution
  *
  * @param sigma The covariance matrix of the base multivariate gaussian.
  * */
case class MESN(
  tau: Double, alpha: DenseVector[Double],
  mu: DenseVector[Double], sigma: DenseMatrix[Double]) extends
  SkewSymmDistribution[DenseVector[Double]](
      basisDistr = MultivariateGaussian(mu + alpha*tau, sigma + alpha*alpha.t),
      warpingDistr = Gaussian(0.0, 1.0),
      cutoff = tau)(
      VectorField(alpha.length)) {

  private lazy val adjustedCenter = mu + alpha*tau

  private lazy val delta = sqrt(1.0 + quadraticForm(cholesky(sigma), alpha))

  override protected val w = DataPipe(
    (x: DenseVector[Double]) => crossQuadraticForm(alpha, cholesky(sigma), x - adjustedCenter)/delta)

  override protected val warped_cutoff: Double = tau*sqrt(1 + quadraticForm(cholesky(sigma), alpha))

  override def draw() = basisDistr.draw() + alpha*(tau + warpingDistr.draw())

}
