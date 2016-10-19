package io.github.mandar2812.dynaml.probability.distributions

import breeze.numerics._

import math.{Pi, log1p}
import breeze.stats.distributions.{ContinuousDistr, Moments, Rand, RandBasis}
import io.github.mandar2812.dynaml.algebra._
import io.github.mandar2812.dynaml.algebra.PartitionedMatrixOps._

import scala.runtime.ScalaRunTime

/**
  * Represents a Gaussian distribution over a single real variable.
  *
  * @author dlwh
  */
case class BlockedMultiVariateGaussian(mean: PartitionedVector,
                                       covariance: PartitionedPSDMatrix)(implicit rand: RandBasis = Rand)
  extends ContinuousDistr[PartitionedVector] with Moments[PartitionedVector, PartitionedPSDMatrix] {
  def draw() = {
    val nE: Int = (mean.rows/mean.rowBlocks).toInt
    val z: PartitionedVector = PartitionedVector(mean.rows, nE, (c) => rand.gaussian(0, 1).draw())
    val m: PartitionedVector = root * z
    m + mean
  }

  private val root: LowerTriPartitionedMatrix = bcholesky(covariance)

  override def toString() =  ScalaRunTime._toString(this)

  override def unnormalizedLogPdf(t: PartitionedVector) = {
    val centered: PartitionedVector = t - mean
    val z: PartitionedVector = root \ centered
    val slv: PartitionedVector = root.t \ z
    val d: Double = slv dot centered
    -1.0*d/2.0

  }

  override lazy val logNormalizer = {
    // determinant of the cholesky decomp is the sqrt of the determinant of the cov matrix
    // this is the log det of the cholesky decomp
    val det = bsum(blog(bdiag(root)))
    mean.rows.toDouble/2 *  log(2 * Pi) + det
  }

  def variance = covariance
  def mode = mean
  lazy val entropy = {
    mean.rows.toDouble * log1p(2 * Pi) + bsum(blog(bdiag(root)))
  }
}

