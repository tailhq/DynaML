package io.github.mandar2812.dynaml.probability

import breeze.stats.distributions.Density
import io.github.mandar2812.dynaml.pipes.DataPipe

import scala.util.Random

/**
  * Created by mandar on 26/7/16.
  */

class ProbabilityModel[
ConditioningSet, Domain,
Dist <: Density[ConditioningSet],
DistL <: Density[Domain]](
  p: RandomVarWithDistr[ConditioningSet, Dist],
  c: DataPipe[ConditioningSet, RandomVarWithDistr[Domain, DistL]])
  extends RandomVarWithDistr[(ConditioningSet, Domain), Density[(ConditioningSet, Domain)]] {

  val prior: RandomVarWithDistr[ConditioningSet, Dist] = p

  val likelihood: DataPipe[ConditioningSet, RandomVarWithDistr[Domain, DistL]] = c

  var Max_Candidates: Int = 10000

  override val underlyingDist = new Density[(ConditioningSet, Domain)] {
    override def apply(x: (ConditioningSet, Domain)): Double =
      prior.underlyingDist(x._1)*likelihood(x._1).underlyingDist(x._2)
  }

  val sample = prior.sample >
    DataPipe(
      (c: ConditioningSet) => (c, likelihood(c).sample())
    )

  val posterior = DataPipe((data: Domain) => {

    val sampl = this.prior.sample
    val q = this.prior.underlyingDist

    val M = (1 to Max_Candidates).map(i => {
      likelihood(sampl()).underlyingDist(data)
    }).sum/Max_Candidates.toDouble

    val postD = new Density[ConditioningSet] {
      override def apply(x: ConditioningSet): Double =
        prior.underlyingDist(x)*likelihood(x).underlyingDist(data)
    }

    new RandomVarWithDistr[ConditioningSet, Density[ConditioningSet]] {

      override val underlyingDist: Density[ConditioningSet] = postD

      override val sample: DataPipe[Unit, ConditioningSet] = DataPipe(() => {
        val iterations = 0
        var accepted = false
        var accepted_sample: ConditioningSet = sampl()

        while(!accepted && iterations < Max_Candidates) {
          // generate a candidate
          val candidate = sampl()
          val a = underlyingDist(candidate)/(M*q(candidate))
          if(Random.nextDouble() <= a) {
            accepted = true
            accepted_sample = candidate
          }
        }

        accepted_sample
      })
    }

  })

}

object ProbabilityModel {
  def apply[
  ConditioningSet, Domain,
  Dist <: Density[ConditioningSet],
  DistL <: Density[Domain]
  ](p: RandomVarWithDistr[ConditioningSet, Dist],
    c: DataPipe[ConditioningSet, RandomVarWithDistr[Domain, DistL]]) = new ProbabilityModel(p,c)
}