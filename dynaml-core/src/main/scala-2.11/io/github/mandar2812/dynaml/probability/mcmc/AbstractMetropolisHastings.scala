package io.github.mandar2812.dynaml.probability.mcmc

import spire.algebra.Field
import breeze.stats.distributions.{Density, Rand, RandBasis}
import breeze.stats.mcmc.BaseMetropolisHastings

/**
  * Created by mandar on 06/01/2017.
  */
case class AbstractMetropolisHastings[T](
  logLikelihood: T => Double, proposalStep: Density[T] with Rand[T],
  init: T, burnIn: Long = 0, dropCount: Int = 0)(
  implicit rand:RandBasis=Rand, f: Field[T]) extends
  BaseMetropolisHastings[T](logLikelihood, init, burnIn, dropCount)(rand) {

  def proposalDraw(x: T): T = f.plus(proposalStep.draw(),x)


  def observe(x: T) = this.copy(burnIn=0, init = x)

  override def logTransitionProbability(start: T, end: T) = proposalStep.logApply(f.minus(end, start))
}
