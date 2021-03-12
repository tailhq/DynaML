package io.github.tailhq.dynaml.probability.distributions

import breeze.numerics.log
import breeze.stats.distributions.{ContinuousDistr, HasCdf}
import io.github.tailhq.dynaml.pipes.DataPipe
import spire.algebra.Field

/**
  *
  * A generalised skew symmetric distribution has the following components
  *
  * <ul>
  *   <li>&phi;: A symmetric distribution acting as the base</li>
  *   <li>G: A symmetric distribution acting as the warping function</li>
  *   <li>w: An odd function from the domain type [[T]] to [[Double]].</li>
  *   <li>&tau;: Cutoff</li>
  * </ul>
  *
  * The probability density of a
  * generalised skew symmetric distribution is
  *
  * &rho;(x) = &phi;(x)*G(w(x) + &tau;)
  *
  * @author tailhq date: 04/01/2017.
  *
  * */
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
