package io.github.mandar2812.dynaml.probability.distributions

import breeze.linalg.{DenseVector, sum}
import breeze.stats.distributions.{ContinuousDistr, Multinomial}

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