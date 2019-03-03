package io.github.mandar2812.dynaml.probability

import io.github.mandar2812.dynaml.pipes.DataPipe

/**
  * Created by mandar on 09/01/2017.
  */
trait JointProbabilityScheme[
Domain1, Domain2,
P <: RandomVariable[Domain1], D <: RandomVariable[Domain2]]
  extends RandomVariable[(Domain1, Domain2)] {

  val prior: P

  val likelihood: DataPipe[Domain1, D]

  /**
    * Generate a sample from
    * the random variable
    *
    **/
  override val sample = DataPipe(() => {
    val ps = prior.sample.run()
    (ps, likelihood(ps).sample.run())
  })
}

trait BayesJointProbabilityScheme[Domain1, Domain2,
P <: RandomVariable[Domain1],
D <: RandomVariable[Domain2]]
  extends JointProbabilityScheme[Domain1, Domain2, P, D] {

  val posterior: DataPipe[Domain2, RandomVariable[Domain1]]
}