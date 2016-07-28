package io.github.mandar2812.dynaml.probability

import breeze.stats.distributions.Density
import io.github.mandar2812.dynaml.pipes.DataPipe

/**
  * Created by mandar on 26/7/16.
  */

class ProbabilityModel[ConditioningSet, Domain, Dist <: Density[ConditioningSet], DistL <: Density[Domain]](
  p: RandomVarWithDistr[ConditioningSet, Dist],
  c: DataPipe[ConditioningSet, RandomVarWithDistr[Domain, DistL]])
  extends RandomVarWithDistr[(ConditioningSet, Domain), Density[(ConditioningSet, Domain)]] {

  val likelihood: DataPipe[ConditioningSet, RandomVariable[Domain] with HasDistribution[Domain]] = c

  val prior: RandomVariable[ConditioningSet] with HasDistribution[ConditioningSet] = p

  override val underlyingDist = new Density[(ConditioningSet, Domain)] {
    override def apply(x: (ConditioningSet, Domain)): Double =
      prior.underlyingDist(x._1)*likelihood(x._1).underlyingDist(x._2)
  }

  val sample = prior.sample >
    DataPipe(
      (c: ConditioningSet) => (c, likelihood(c).sample())
    )

}

object ProbabilityModel {
  def apply[
  ConditioningSet, Domain,
  Dist <: Density[ConditioningSet],
  DistL <: Density[Domain]
  ](p: RandomVarWithDistr[ConditioningSet, Dist],
    c: DataPipe[ConditioningSet, RandomVarWithDistr[Domain, DistL]]) = new ProbabilityModel(p,c)
}