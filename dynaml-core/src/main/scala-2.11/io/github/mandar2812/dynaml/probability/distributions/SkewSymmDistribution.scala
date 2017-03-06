package io.github.mandar2812.dynaml.probability.distributions

import breeze.numerics.log
import breeze.stats.distributions.{ContinuousDistr, HasCdf}
import io.github.mandar2812.dynaml.pipes.DataPipe
import spire.algebra.Field

/**
  * @author mandar2812 date: 04/01/2017.
  *
  * A generalised skew symmetric distribution has three components
  *
  * basisDistr: A symmetric distribution acting as the base
  * warpingDistr: A symmetric distribution acting as the warping function
  * w: An odd function from the domain type [[T]] to [[Double]].
  */
abstract class SkewSymmDistribution[T](implicit f: Field[T])
  extends ContinuousDistr[T] {

  protected val basisDistr: ContinuousDistr[T]

  protected val warpingDistr: ContinuousDistr[Double] with HasCdf

  protected val cutoff: Double = 0.0

  protected val w: DataPipe[T, Double]

  lazy private val p_cutoff: Double = warpingDistr.cdf(cutoff)

  /**
    * The warped cutoff is an adjusted value of the
    * cutoff based on the cutoff value and skewness parameters.
    * To be implemented by extending class.
    * */
  protected val warped_cutoff: Double

  override def unnormalizedLogPdf(x: T) =
    basisDistr.unnormalizedLogPdf(x) + log(warpingDistr.cdf(w(x) + warped_cutoff))

  override def logNormalizer =
    log(p_cutoff) + basisDistr.logNormalizer

  override def draw() = {
    //sample X ~ warpingDist and Y ~ baseDist independently
    //return Y if X < w(Y) else return -Y
    val x = warpingDistr.draw()
    val y = basisDistr.draw()

    if (x < w(y) + cutoff) y else f.negate(y)
  }
}
