package io.github.mandar2812.dynaml.probability

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.probability.distributions.MixtureDistribution


/**
  * Top level class for representing
  * random variable mixtures.
  *
  * @author mandar2812 date 14/06/2017.
  * */
trait MixtureRV[Domain, BaseRV <: RandomVariable[Domain]] extends
  RandomVariable[Domain] {

  val mixture_selector: MultinomialRV

  val components: Seq[BaseRV]

}

object MixtureRV {

  def apply[Domain, BaseRV <: RandomVariable[Domain]](
    distributions: Seq[BaseRV],
    weights: DenseVector[Double]): MixtureRV[Domain, BaseRV] =
    new MixtureRV[Domain, BaseRV] {

      override val mixture_selector = MultinomialRV(weights)

      override val components = distributions

      val sample = mixture_selector.sample > DataPipe((i: Int) => components(i).draw)
    }

}

/**
  * A random variable mixture over a continuous domain
  * @author mandar2812 date 14/06/2017
  * */
trait ContinuousMixtureRV[Domain, BaseRV <: ContinuousRandomVariable[Domain]] extends
  ContinuousRandomVariable[Domain] with
  MixtureRV[Domain, BaseRV]

/**
  * A random variable mixture over a continuous domain,
  * having a computable probability distribution
  * @author mandar2812 date 14/06/2017
  * */
class ContinuousDistrMixture[
Domain, BaseRV <: ContinuousDistrRV[Domain]](
  distributions: Seq[BaseRV],
  selector: MultinomialRV) extends
  ContinuousMixtureRV[Domain, BaseRV] with
  ContinuousDistrRV[Domain] {

  override val mixture_selector = selector

  override val components = distributions

  override val underlyingDist = MixtureDistribution(
    components.map(_.underlyingDist),
    selector.underlyingDist.params)
}

object ContinuousDistrMixture {

  def apply[Domain, BaseRV <: ContinuousDistrRV[Domain]](
    distributions: Seq[BaseRV],
    selector: DenseVector[Double]): ContinuousDistrMixture[Domain, BaseRV] =
    new ContinuousDistrMixture(distributions, MultinomialRV(selector))
}
