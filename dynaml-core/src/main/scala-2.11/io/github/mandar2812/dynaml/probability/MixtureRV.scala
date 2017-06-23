package io.github.mandar2812.dynaml.probability

import spire.algebra.{Field, VectorSpace}
import breeze.linalg.DenseVector
import breeze.stats.distributions.{ContinuousDistr, Moments}
import io.github.mandar2812.dynaml.algebra.{PartitionedPSDMatrix, PartitionedVector}
import io.github.mandar2812.dynaml.analysis.PartitionedVectorField
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.probability.distributions.{
BlockedMultiVariateGaussian, HasErrorBars,
MixtureDistribution, MixtureWithConfBars}


/**
  * Top level class for representing
  * random variable mixtures.
  *
  * @tparam Domain Domain over which each mixture
  *                component is defined
  *
  * @tparam BaseRV The type of each mixture component, must
  *                be a sub type of [[RandomVariable]]
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
  *
  * @tparam Domain Domain over which each mixture
  *                component is defined
  *
  * @tparam BaseRV The type of each mixture component, must
  *                be a sub type of [[ContinuousRandomVariable]]
  *
  * @author mandar2812 date 14/06/2017
  * */
trait ContinuousMixtureRV[Domain, BaseRV <: ContinuousRandomVariable[Domain]] extends
  ContinuousRandomVariable[Domain] with
  MixtureRV[Domain, BaseRV]

/**
  * A random variable mixture over a continuous domain,
  * having a computable probability distribution
  *
  * @tparam Domain Domain over which each mixture
  *                component is defined
  *
  * @tparam BaseRV The type of each mixture component, must
  *                be a sub type of [[ContinuousRVWithDistr]]
  *
  * @author mandar2812 date 14/06/2017
  * */
class ContinuousDistrMixture[Domain, BaseRV <: ContinuousRVWithDistr[Domain, ContinuousDistr[Domain]]](
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

/**
  * A random variable mixture over a continuous domain,
  * having a computable probability distribution and ability
  * to generate error bars/confidence intervals.
  *
  * @tparam Domain Domain over which each mixture
  *                component is defined
  *
  * @tparam Var The type of second moment i.e. variance of the random variables
  *             defined over [[Domain]].
  *
  * @tparam BaseDistr The type of each mixture component distribution,
  *                   must be a continuous breeze distribution with
  *                   defined moments and implement [[HasErrorBars]]
  *
  * @author mandar2812 date 14/06/2017
  * */
private[dynaml] class ContMixtureRVBars[Domain, Var,
BaseDistr <: ContinuousDistr[Domain] with Moments[Domain, Var] with HasErrorBars[Domain]](
  distributions: Seq[BaseDistr],
  selector: MultinomialRV)(
  implicit v: VectorSpace[Domain, Double]) extends
  ContinuousDistrMixture[Domain, ContinuousRVWithDistr[Domain, ContinuousDistr[Domain]]](
    distributions.map(RandomVariable(_)), selector) {

  override val underlyingDist = MixtureWithConfBars(distributions, selector.underlyingDist.params)
}

/**
  * Contains convenience methods for creating
  * various mixture random variable instances.
  * */
object ContinuousDistrMixture {

  def apply[Domain, BaseRV <: ContinuousDistrRV[Domain]](
    distributions: Seq[BaseRV],
    selector: DenseVector[Double]): ContinuousDistrMixture[Domain, BaseRV] =
    new ContinuousDistrMixture(distributions, MultinomialRV(selector))

  def apply[Domain, Var, BaseDistr <: ContinuousDistr[Domain] with Moments[Domain, Var] with HasErrorBars[Domain]](
    distributions: Seq[BaseDistr], selector: DenseVector[Double])(
    implicit v: VectorSpace[Domain, Double]) =
    new ContMixtureRVBars[Domain, Var, BaseDistr](distributions, MultinomialRV(selector))

  def apply(num_elements_per_block: Int)(
    distributions: Seq[BlockedMultiVariateGaussian],
    selector: DenseVector[Double]) = {

    val num_dim = distributions.head.mean.rows

    implicit val ev = PartitionedVectorField(num_dim, num_elements_per_block)

    apply[PartitionedVector, PartitionedPSDMatrix, BlockedMultiVariateGaussian](distributions, selector)
  }

}
