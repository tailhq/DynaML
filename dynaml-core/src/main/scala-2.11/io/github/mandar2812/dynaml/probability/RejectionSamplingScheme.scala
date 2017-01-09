package io.github.mandar2812.dynaml.probability

import spire.algebra.Field
import breeze.stats.distributions.{ContinuousDistr, Density, Rand}
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.probability.distributions.GenericDistribution

import scala.util.Random
import org.apache.log4j.Logger


/**
  * Created by mandar on 26/7/16.
  */

abstract class RejectionSamplingScheme[
ConditioningSet, Domain,
Dist <: Density[ConditioningSet] with Rand[ConditioningSet],
DistL <: Density[Domain] with Rand[Domain],
JointDist <: Density[(ConditioningSet, Domain)] with Rand[(ConditioningSet, Domain)]](
  p: RandomVarWithDistr[ConditioningSet, Dist],
  c: DataPipe[ConditioningSet, RandomVarWithDistr[Domain, DistL]])
  extends RandomVarWithDistr[(ConditioningSet, Domain), JointDist]
    with BayesJointProbabilityScheme[
    ConditioningSet, Domain,
    RandomVarWithDistr[ConditioningSet, Dist],
    RandomVarWithDistr[Domain, DistL]] { self =>

  override val prior: RandomVarWithDistr[ConditioningSet, Dist] = p

  override val likelihood: DataPipe[ConditioningSet, RandomVarWithDistr[Domain, DistL]] = c

  var Max_Candidates: Int = 1000

  var Max_Estimations: Int = 10000

  override val sample = prior.sample >
    DataPipe[ConditioningSet, ConditioningSet, Domain](
      (c: ConditioningSet) => (c, likelihood(c).sample())
    )

  override val posterior: DataPipe[Domain, RandomVariable[ConditioningSet]] =
    DataPipe((data: Domain) => {

    val sampl = this.prior.sample
    val q = this.prior.underlyingDist

    val M = (1 to Max_Estimations).map(i => {
      likelihood(sampl()).underlyingDist(data)
    }).sum/Max_Estimations.toDouble

    new RandomVarWithDistr[ConditioningSet, GenericDistribution[ConditioningSet]] { innerself =>

      val logger = Logger.getLogger(this.getClass)

      override val sample: DataPipe[Unit, ConditioningSet] = DataPipe(() => {
        val iterations = 0
        var accepted = false
        var accepted_sample: ConditioningSet = sampl()

        while(!accepted && iterations < Max_Candidates) {
          // generate a candidate
          val candidate = sampl()
          val a = underlyingDist(candidate)/(M*q(candidate))
          if(Random.nextDouble() <= a) {
            logger.info("... Sample Accepted ...")
            accepted = true
            accepted_sample = candidate
          }
        }

        accepted_sample
      })

      override val underlyingDist: GenericDistribution[ConditioningSet] = new GenericDistribution[ConditioningSet] {
        override def apply(x: ConditioningSet): Double =
          prior.underlyingDist(x)*likelihood(x).underlyingDist(data)

        override def draw() = innerself.sample()
      }

    }
  })

}

object RejectionSamplingScheme {
  def apply[ConditioningSet, Domain](
    p: ContinuousDistrRV[ConditioningSet], c: DataPipe[ConditioningSet, ContinuousDistrRV[Domain]],
    proposalDist: RandomVarWithDistr[ConditioningSet, ContinuousDistr[ConditioningSet]])(
     implicit vectorSpace: Field[ConditioningSet]) =
    new ContinuousMCMC(p, c, proposalDist)
}