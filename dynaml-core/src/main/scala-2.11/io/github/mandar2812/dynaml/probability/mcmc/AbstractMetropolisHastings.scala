package io.github.mandar2812.dynaml.probability.mcmc

import spire.algebra.Field
import breeze.stats.distributions.{Density, Rand, RandBasis}
import breeze.stats.mcmc.{BaseMetropolisHastings, SymmetricMetropolisHastings}

/**
  * Created by mandar on 06/01/2017.
  */
case class AbstractMetropolisHastings[T, Dist <: Density[T] with Rand[T]](
  logLikelihood: T => Double, proposalStep: Dist,
  init: T, burnIn: Long = 0, dropCount: Int = 0)(
  implicit rand:RandBasis=Rand, f: Field[T]) extends
  BaseMetropolisHastings[T](logLikelihood, init, burnIn, dropCount)(rand)
  with SymmetricMetropolisHastings[T] { self =>

  def proposalDraw(x: T): T = f.plus(proposalStep.draw(),x)

  val likelihood = logLikelihood

  val proposal = proposalStep


  def observe(x: T) = this.copy(likelihood, proposal, burnIn = 0L, init = x)

}
