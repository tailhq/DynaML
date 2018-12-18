package io.github.mandar2812.dynaml.probability.distributions

import breeze.linalg.{DenseVector, sum}
import breeze.stats.distributions.{ContinuousDistr, Moments, Multinomial}
import spire.algebra.VectorSpace

/**
  * Distribution consisting of a mixture of components
  * and a probability weight over each component.
  *
  * @param distributions An array of mixture components,
  *                      each one a breeze continuous distribution
  * @param probabilities A multinomial distribution having the probabilities of
  *                      each component.
  * @author mandar2812 date 14/06/2017.
  * */
class MixtureDistribution[I](
  distributions: Seq[ContinuousDistr[I]],
  probabilities: Multinomial[DenseVector[Double], Int]) extends
  AbstractContinuousDistr[I] {

  val num_components = probabilities.params.length

  protected val multinomial: Multinomial[DenseVector[Double], Int] = probabilities

  def components = distributions

  override def unnormalizedLogPdf(x: I) = math.log(
    sum(multinomial.params.mapPairs((i,p) => p*components(i).pdf(x)))
  )

  override def logNormalizer = 0d

  override def draw() = components(multinomial.draw()).draw()
}


object MixtureDistribution {

  /**
    * Convenience method for creating a mixture distribution.
    * */
  def apply[I](
    distributions: Seq[ContinuousDistr[I]],
    weights: DenseVector[Double]): MixtureDistribution[I] =
    new MixtureDistribution(distributions, new Multinomial[DenseVector[Double], Int](weights))
}


/**
  * A mixture distribution which can compute its mean
  * and generate confidence intervals.
  * @tparam I The domain of the random variable
  * @tparam V The type of the variance returned by the mixture components
  * @author mandar2812 date 15/06/2017
  * */
class MixtureWithConfBars[I, V](
  distributions: Seq[ContinuousDistr[I] with Moments[I, V] with HasErrorBars[I]],
  probabilities: Multinomial[DenseVector[Double], Int])(
  implicit vI: VectorSpace[I, Double]) extends
  MixtureDistribution[I](distributions, probabilities) with
  HasErrorBars[I] {

  private val weightsArr = probabilities.params.toArray

  override def confidenceInterval(s: Double) =
    distributions.zip(weightsArr).map(c => {
      val (lower, upper) = c._1.confidenceInterval(s)

      (vI.timesr(lower, c._2), vI.timesr(upper, c._2))
    }).reduce((a,b) =>
      (vI.plus(a._1, b._1), vI.plus(a._2, b._2))
    )


  def mean = distributions.zip(weightsArr)
    .map(c => vI.timesr(c._1.mean, c._2))
    .reduce((a,b) => vI.plus(a,b))

}

object MixtureWithConfBars {

  def apply[I, V](
    distributions: Seq[ContinuousDistr[I] with Moments[I, V] with HasErrorBars[I]],
    weights: DenseVector[Double])(
    implicit vI: VectorSpace[I, Double]): MixtureWithConfBars[I, V] =
    new MixtureWithConfBars(distributions, new Multinomial[DenseVector[Double], Int](weights))
}
