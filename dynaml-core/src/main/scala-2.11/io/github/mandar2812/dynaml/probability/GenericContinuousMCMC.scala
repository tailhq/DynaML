/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
* */
package io.github.mandar2812.dynaml.probability

import spire.algebra.Field
import breeze.stats.distributions.ContinuousDistr
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.probability.distributions.AbstractContinuousDistr
import io.github.mandar2812.dynaml.probability.mcmc.GeneralMetropolisHastings

/**
  * Monte Carlo based bayesian inference model where the parameter
  * space is known to be continuous and hence represented via
  * a [[ContinuousDistrRV]] instance.
  *
  * @tparam ConditioningSet The type representing the model parameters
  * @tparam Domain The type representing the observed data.
  * @param p The prior distribution on model parameters as a DynaML random variable
  * @param c The likelihood of the data given a particular value of parameters
  *
  * @author mandar2812 date 06/01/2017
  *
  * */
class GenericContinuousMCMC[ConditioningSet, Domain](
  p: ContinuousRVWithDistr[ConditioningSet, ContinuousDistr[ConditioningSet]],
  c: DataPipe[ConditioningSet, ContinuousRVWithDistr[Domain, ContinuousDistr[Domain]]],
  proposalDist: ContinuousRVWithDistr[ConditioningSet, ContinuousDistr[ConditioningSet]],
  burnIn: Long = 1000L, dropCount: Int = 100)(implicit vectorSpace: Field[ConditioningSet])
  extends RejectionSamplingScheme[
    ConditioningSet, Domain, ContinuousDistr[ConditioningSet], ContinuousDistr[Domain],
    AbstractContinuousDistr[(ConditioningSet, Domain)]](p, c) {


  override val underlyingDist = new AbstractContinuousDistr[(ConditioningSet, Domain)] {

    private val priorSample = prior.sample.run()

    override def unnormalizedLogPdf(x: (ConditioningSet, Domain)) =
      prior.underlyingDist.unnormalizedLogPdf(x._1) +
        likelihood(x._1).underlyingDist.unnormalizedLogPdf(x._2)

    override def logNormalizer = prior.underlyingDist.logNormalizer +
      likelihood(priorSample).underlyingDist.logNormalizer

    override def draw() = {
      val ps = prior.underlyingDist.draw()
      (ps, likelihood(ps).underlyingDist.draw())
    }
  }

  override val posterior: DataPipe[Domain, RandomVariable[ConditioningSet]] = DataPipe((data: Domain) => {

    val logLikelihoodFunc = (candidate: ConditioningSet) => {
      prior.underlyingDist.logPdf(candidate) + likelihood(candidate).underlyingDist.logPdf(data)
    }

    //Initialize an MCMC sampler
    val sampler = GeneralMetropolisHastings(
      LikelihoodModel(logLikelihoodFunc), proposalDist.underlyingDist,
      prior.sample.run(), burnIn, dropCount)

    RandomVariable(sampler.draw _)
  })
}


