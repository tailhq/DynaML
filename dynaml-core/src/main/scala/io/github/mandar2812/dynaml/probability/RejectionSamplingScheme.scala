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

import breeze.stats.distributions.{Density, Rand}
import io.github.mandar2812.dynaml.pipes._
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
JointDist <: Density[(ConditioningSet, Domain)] with Rand[(ConditioningSet, Domain)],
Likelihood <: RandomVarWithDistr[Domain, DistL]](
  p: RandomVarWithDistr[ConditioningSet, Dist],
  c: DataPipe[ConditioningSet, Likelihood])
  extends RandomVarWithDistr[(ConditioningSet, Domain), JointDist]
    with BayesJointProbabilityScheme[
    ConditioningSet, Domain,
    RandomVarWithDistr[ConditioningSet, Dist],
    Likelihood] { self =>

  override val prior: RandomVarWithDistr[ConditioningSet, Dist] = p

  override val likelihood: DataPipe[ConditioningSet, Likelihood] = c

  var Max_Candidates: Int = 1000

  var Max_Estimations: Int = 10000

  override val sample = prior.sample >
  BifurcationPipe[ConditioningSet, ConditioningSet, Domain](
      (c: ConditioningSet) => (c, likelihood(c).draw)
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
