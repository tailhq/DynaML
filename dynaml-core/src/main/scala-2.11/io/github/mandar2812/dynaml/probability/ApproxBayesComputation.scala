package io.github.mandar2812.dynaml.probability

import io.github.mandar2812.dynaml.pipes.DataPipe
import spire.implicits.cfor

/**
  * @author mandar2812 date 09/01/2017.
  *
  * Abstract representation of the Approximate Bayesian Computation (ABC)
  * scheme.
  */
class ApproxBayesComputation[ConditioningSet, Domain](
  p: RandomVariable[ConditioningSet],
  c: DataPipe[ConditioningSet, RandomVariable[Domain]],
  m: (Domain, Domain) => Double) extends
BayesJointProbabilityScheme[
  ConditioningSet, Domain,
  RandomVariable[ConditioningSet],
  RandomVariable[Domain]] {

  protected var tolerance: Double = 1E-4

  def tolerance_(t: Double) = tolerance = t

  protected var MAX_ITERATIONS: Int = 10000

  def max_iterations_(it: Int) = MAX_ITERATIONS = it

  override val prior = p

  override val likelihood = c

  val metric: (Domain, Domain) => Double = m

  val acceptance: (Domain, Domain) => Boolean = (x, y) => metric(x,y) <= tolerance

  override val posterior: DataPipe[Domain, RandomVariable[ConditioningSet]] =
    DataPipe((data: Domain) => {
      //Generate Sample from prior
      //If acceptance condition is met

      val sampler = () => {
        var candidate = prior.sample()
        var generatedData = likelihood(candidate).sample()

        cfor(1)(count =>
          count < MAX_ITERATIONS && !acceptance(data, generatedData),
          count => count + 1)(count => {
          candidate = prior.sample()
          generatedData = likelihood(candidate).sample()
        })
        candidate
      }
      RandomVariable(sampler)
    })
}

object ApproxBayesComputation {
  def apply[ConditioningSet, Domain](
    p: RandomVariable[ConditioningSet],
    c: DataPipe[ConditioningSet, RandomVariable[Domain]],
    m: (Domain, Domain) => Double) = new ApproxBayesComputation(p, c, m)
}