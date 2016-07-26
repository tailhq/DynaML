package io.github.mandar2812.dynaml.probability

import io.github.mandar2812.dynaml.pipes.DataPipe

/**
  * Created by mandar on 26/7/16.
  */

class ProbabilityModel[ConditioningSet, Domain,
R1 <: RandomVariable[ConditioningSet],
R2 <: RandomVariable[Domain]](p: R1, c: DataPipe[ConditioningSet, R2])
  extends RandomVariable[(ConditioningSet, Domain)] {

  val conditionalRV: DataPipe[ConditioningSet, R2] = c

  val priorRV: R1 = p

  val sample = priorRV.sample >
    DataPipe(
      (c: ConditioningSet) => (c, conditionalRV(c).sample())
    )

}
