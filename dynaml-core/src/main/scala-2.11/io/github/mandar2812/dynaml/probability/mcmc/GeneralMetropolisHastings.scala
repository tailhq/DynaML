package io.github.mandar2812.dynaml.probability.mcmc

import spire.algebra.Field
import breeze.stats.distributions.{ContinuousDistr, Rand, RandBasis}
import breeze.stats.mcmc.BaseMetropolisHastings
import io.github.mandar2812.dynaml.probability.{DifferentiableLikelihoodModel, LikelihoodModel}

/**
  * @author mandar2812 date 06/01/2017.
  * */
case class GeneralMetropolisHastings[T](
  logLikelihoodF: LikelihoodModel[T], proposalStep: ContinuousDistr[T],
  init: T, burnIn: Long = 0, dropCount: Int = 0)(
  implicit rand:RandBasis=Rand, f: Field[T]) extends
  BaseMetropolisHastings[T](logLikelihoodF.run, init, burnIn, dropCount)(rand) { self =>

  override def proposalDraw(x: T): T = f.plus(proposalStep.draw(), x)

  val proposal = proposalStep

  def observe(x: T) = this.copy(logLikelihoodF, proposal, burnIn = 0L, init = x)

  override def logTransitionProbability(start: T, end: T) = 0.0
}

abstract class HamiltonianMetropolis[T](
  logLikelihoodF: DifferentiableLikelihoodModel[T],
  proposalDistr: ContinuousDistr[T], init: T, burnIn: Long = 0)(
  implicit rand:RandBasis=Rand, f: Field[T]) extends
  GeneralMetropolisHastings[T](
    logLikelihoodF, proposalDistr,
    init, burnIn, dropCount = 0) {

  override def proposalDraw(x: T): T = f.plus(proposalStep.draw(), x)

  override val proposal = proposalDistr

  override def observe(x: T) = this.copy(logLikelihoodF, proposal, burnIn = 0L, init = x)

  override def logTransitionProbability(start: T, end: T) = 0.0


}