package io.github.mandar2812.dynaml.probability

import io.github.mandar2812.dynaml.pipes.DataPipe

/**
  * Created by mandar on 26/7/16.
  */

class ProbabilityModel[ConditioningSet, Domain,
R1[ConditioningSet] <: RandomVariable[ConditioningSet],
R2[Domain] <: RandomVariable[Domain]](p: R1[ConditioningSet], c: DataPipe[ConditioningSet, R2[Domain]])
  extends RandomVariable[(ConditioningSet, Domain)] {

  val conditionalRV: DataPipe[ConditioningSet, R2[Domain]] = c

  val priorRV: R1[ConditioningSet] = p

  val sample = priorRV.sample >
    DataPipe(
      (c: ConditioningSet) => (c, conditionalRV(c).sample())
    )

}

object ProbabilityModel {
  def apply[ConditioningSet, Domain,
  R1[ConditioningSet] <: RandomVariable[ConditioningSet],
  R2[Domain] <: RandomVariable[Domain]](
  p: R1[ConditioningSet],
  c: DataPipe[ConditioningSet, R2[Domain]]) = new ProbabilityModel(p,c)
}