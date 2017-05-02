package io.github.mandar2812.dynaml.probability.distributions

import breeze.linalg.DenseVector
import breeze.numerics._

import math.Pi
import breeze.stats.distributions._
import io.github.mandar2812.dynaml.algebra._
import io.github.mandar2812.dynaml.algebra.PartitionedMatrixOps._
import io.github.mandar2812.dynaml.algebra.PartitionedMatrixSolvers._
import io.github.mandar2812.dynaml.probability.RandomVariable
import spire.implicits._

import scala.runtime.ScalaRunTime

/**
  * @author mandar2812 date: 23/10/2016.
  *
  * Multivariate T distribution on [[PartitionedVector]]
  */
case class BlockedMultivariateStudentsT(
  mu: Double,
  mean: PartitionedVector,
  covariance: PartitionedPSDMatrix)(implicit rand: RandBasis = Rand) extends
  AbstractContinuousDistr[PartitionedVector] with
  Moments[PartitionedVector, PartitionedPSDMatrix] with
  HasErrorBars[PartitionedVector] {

  require(mu > 2.0, "Degrees of freedom must be greater than 2.0, for a multivariate t distribution to be defined")

  private val chisq = new ChiSquared(mu)

  def draw() = {
    val w = math.sqrt(mu/chisq.draw())
    val nE: Int = if(mean.rowBlocks > 1L) mean(0L to 0L)._data.head._2.length else mean.rows.toInt
    val z: PartitionedVector = PartitionedVector.rand(mean.rows, nE, RandomVariable(new StudentsT(mu)))*w
    val m: PartitionedVector = root * z
    m + mean
  }

  private lazy val root: LowerTriPartitionedMatrix = bcholesky(covariance)

  override def toString() =  ScalaRunTime._toString(this)

  override def unnormalizedLogPdf(t: PartitionedVector) = {
    val centered: PartitionedVector = t - mean
    val z: PartitionedVector = root \ centered
    val slv: PartitionedVector = root.t \ z

    -0.5*(mu+mean.rows)*log(1.0 + ((slv dot centered) / (mu - 2.0)))

  }

  override lazy val logNormalizer = {
    // determinant of the cholesky decomp is the sqrt of the determinant of the cov matrix
    // this is the log det of the cholesky decomp
    val det = bsum(blog(bdiag(root)))
    ((mean.rows/2) * (log(mu - 2.0) + log(Pi))) + det + lgamma(mu/2.0) - lgamma((mu+mean.rows)/2.0)
  }

  def variance = new PartitionedPSDMatrix(
    covariance._underlyingdata.map(c => (c._1, c._2*(mu/(mu-2.0)))),
    covariance.rows, covariance.cols, covariance.rowBlocks, covariance.colBlocks)

  def mode = mean

  //TODO: Check and correct calculation of entropy for Mult Students T
  lazy val entropy = {
    bsum(blog(bdiag(root))) + (mean.rows/2.0)*log(mu*Pi) + lbeta(mean.rows/2.0, mu/2.0) - lgamma(mean.rows/2.0) +
      (digamma((mu+mean.rows)/2.0) - digamma(mu/2.0))*(mu+mean.rows)/2.0
  }

  override def confidenceInterval(s: Double) = {

    val signFlag = if(s < 0) -1.0 else 1.0
    val nE: Int = if(mean.rowBlocks > 1L) mean(0L to 0L)._data.head._2.length else mean.rows.toInt
    val ones = PartitionedVector.ones(mean.rows, nE)
    val multiplier = signFlag*s

    val bar: PartitionedVector = root*(ones*(multiplier*math.sqrt(mu/(mu-2.0))))

    (mean - bar, mean + bar)

  }
}


