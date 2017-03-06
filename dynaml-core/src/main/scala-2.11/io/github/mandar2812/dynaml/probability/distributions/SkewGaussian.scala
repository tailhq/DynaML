package io.github.mandar2812.dynaml.probability.distributions


import breeze.linalg.{DenseMatrix, DenseVector, cholesky}
import breeze.numerics.sqrt
import breeze.stats.distributions._
import io.github.mandar2812.dynaml.analysis.{PartitionedVectorField, VectorField}
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.utils._
import io.github.mandar2812.dynaml.algebra._
import spire.implicits._
import io.github.mandar2812.dynaml.algebra.PartitionedMatrixOps._
import io.github.mandar2812.dynaml.probability.{E, MeasurableFunction, RandomVariable}
import io.github.mandar2812.dynaml.utils

/**
  * @author mandar2812 date: 02/01/2017.
  *
  * Univariate skew gaussian distribution
  */
case class SkewGaussian(
  alpha: Double, mu: Double = 0.0,
  sigma: Double = 1.0) extends
  SkewSymmDistribution[Double] {

  override protected val basisDistr: ContinuousDistr[Double] = Gaussian(mu, sigma)

  override protected val warpingDistr: ContinuousDistr[Double] with HasCdf = Gaussian(0.0, 1.0)

  override protected val w = DataPipe((x: Double) => alpha*(x-mu)/sigma)

  override protected val warped_cutoff: Double = 0.0
}

/**
  * The univariate extended skew gaussian distribution
  * */
case class ExtendedSkewGaussian(
  alpha0: Double, alpha: Double,
  mu: Double = 0.0, sigma: Double = 1.0)(
  implicit rand: RandBasis = Rand)
  extends SkewSymmDistribution[Double] {



  override protected val basisDistr: ContinuousDistr[Double] = Gaussian(mu, sigma)

  override protected val warpingDistr: ContinuousDistr[Double] with HasCdf = Gaussian(0.0, 1.0)

  override protected val cutoff: Double = alpha0

  override protected val w = DataPipe((x: Double) => alpha*(x-mu)/sigma)

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
  SkewSymmDistribution[DenseVector[Double]]()(VectorField(alpha.length)) {

  override protected val basisDistr: ContinuousDistr[DenseVector[Double]] = MultivariateGaussian(mu, sigma)

  override protected val warpingDistr: ContinuousDistr[Double] with HasCdf = Gaussian(0.0, 1.0)

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
  SkewSymmDistribution[DenseVector[Double]]()(VectorField(alpha.length)) {

  override protected val basisDistr: ContinuousDistr[DenseVector[Double]] = MultivariateGaussian(mu, sigma)

  override protected val warpingDistr: ContinuousDistr[Double] with HasCdf = Gaussian(0.0, 1.0)

  override protected val cutoff: Double = tau

  private lazy val omega = diagonal(sigma)

  private lazy val sqrtSigma = sqrt(omega)

  private lazy val correlationMatrix: DenseMatrix[Double] = omega\sigma

  override protected val w = DataPipe((x: DenseVector[Double]) => crossQuadraticForm(alpha, cholesky(sqrtSigma), x-mu))

  override protected val warped_cutoff: Double = tau*sqrt(1.0 + quadraticForm(cholesky(correlationMatrix), alpha))
}


/**
  * The univariate version of MESN of Adcock and Schutes.
  *
  * See also: [[MESN]]
  * */
case class UESN(
  tau: Double, alpha: Double,
  mu: Double = 0.0, sigma: Double = 1.0)(
  implicit rand: RandBasis = Rand)
  extends SkewSymmDistribution[Double] {

  override protected val basisDistr: ContinuousDistr[Double] = Gaussian(mu + alpha*tau, sigma + math.abs(alpha))

  override protected val warpingDistr: ContinuousDistr[Double] with HasCdf = Gaussian(0.0, 1.0)

  override protected val cutoff: Double = tau

  private val truncatedGaussian = TruncatedGaussian(tau, 1.0, 0.0, Double.PositiveInfinity)

  private lazy val adjustedCenter = mu + alpha*tau

  private lazy val delta = sqrt(1.0 + alpha*alpha/sigma)

  override protected val w = DataPipe((x: Double) => alpha*(x-adjustedCenter)/(sigma*delta))

  override protected val warped_cutoff: Double = tau*delta

  override def draw() = basisDistr.draw() + alpha*truncatedGaussian.draw()

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
  SkewSymmDistribution[DenseVector[Double]]()(VectorField(alpha.length)) {

  private val truncatedGaussian = TruncatedGaussian(tau, 1.0, 0.0, Double.PositiveInfinity)

  private lazy val adjustedCenter = mu + alpha*tau

  private lazy val rootSigma = cholesky(sigma)

  override protected val cutoff: Double = tau

  override protected val basisDistr: ContinuousDistr[DenseVector[Double]] =
    MultivariateGaussian(mu + alpha*tau, sigma + alpha*alpha.t)

  override protected val warpingDistr: ContinuousDistr[Double] with HasCdf = Gaussian(0.0, 1.0)

  private lazy val delta = sqrt(1.0 + quadraticForm(rootSigma, alpha))

  override protected val w = DataPipe(
    (x: DenseVector[Double]) => crossQuadraticForm(alpha, rootSigma, x - adjustedCenter)/delta)

  override protected val warped_cutoff: Double = tau*delta

  override def draw() = basisDistr.draw() + alpha*truncatedGaussian.draw()

}

/**
  * Extended Multivariate Skew-Gaussian distribution
  * as specified in Adcock and Schutes, defined on [[PartitionedVector]]
  *
  * @param tau Determines the cutoff of the warping function
  *
  * @param alpha A breeze [[DenseVector]] which represents the skewness parameters
  *
  * @param mu The center of the distribution
  *
  * @param sigma The covariance matrix of the base multivariate gaussian.
  * */
case class BlockedMESN(
  tau: Double, alpha: PartitionedVector,
  mu: PartitionedVector, sigma: PartitionedPSDMatrix) extends
  SkewSymmDistribution[PartitionedVector]()(
    PartitionedVectorField(alpha.rows, (alpha.rows/alpha.rowBlocks).toInt))
  with Moments[PartitionedVector, PartitionedPSDMatrix] {

  private val truncatedGaussian = TruncatedGaussian(tau, 1.0, 0.0, Double.PositiveInfinity)

  implicit val f = PartitionedVectorField(alpha.rows, (alpha.rows/alpha.rowBlocks).toInt)

  private lazy val adjustedCenter: PartitionedVector = mu + alpha*tau

  private val varAdjustment = PartitionedPSDMatrix.fromOuterProduct(alpha)

  private lazy val adjustedVariance: PartitionedPSDMatrix = sigma + varAdjustment

  private lazy val rootSigma: LowerTriPartitionedMatrix = bcholesky(sigma)

  private lazy val delta = sqrt(1.0 + blockedQuadraticForm(rootSigma, alpha))

  override protected val basisDistr: ContinuousDistr[PartitionedVector] =
    BlockedMultiVariateGaussian(
      mu + alpha*tau,
      sigma + PartitionedPSDMatrix.fromOuterProduct(alpha))

  override protected val warpingDistr: ContinuousDistr[Double] with HasCdf = Gaussian(0.0, 1.0)

  override protected val cutoff: Double = tau

  override protected val w = DataPipe(
    (x: PartitionedVector) => blockedCrossQuadraticForm(alpha, rootSigma, x - adjustedCenter)/delta)

  override protected val warped_cutoff: Double = tau*sqrt(1 + blockedQuadraticForm(rootSigma, alpha))

  override def draw() = basisDistr.draw() + alpha*truncatedGaussian.draw()

  override def mean = adjustedCenter + alpha*warpingDistr.pdf(tau)

  override def variance = adjustedVariance + varAdjustment*(warpingDistr.pdf(tau)*utils.hermite(1, tau))

  //TODO: Improve calculation of entropy
  override def entropy = Double.NaN//E(MeasurableFunction(RandomVariable(this))(x => math.log(1.0/pdf(x))))

  //TODO: Fix calculation of mode
  override def mode = adjustedCenter
}
