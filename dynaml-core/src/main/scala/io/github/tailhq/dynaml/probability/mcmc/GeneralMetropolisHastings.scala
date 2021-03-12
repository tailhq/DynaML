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
package io.github.tailhq.dynaml.probability.mcmc

import spire.algebra.Field
import breeze.stats.distributions.{ContinuousDistr, Rand, RandBasis}
import breeze.stats.mcmc.BaseMetropolisHastings
import io.github.tailhq.dynaml.probability.{DifferentiableLikelihoodModel, LikelihoodModel}

/**
  * @author tailhq date 06/01/2017.
  * */
case class GeneralMetropolisHastings[T](
  logLikelihoodF: LikelihoodModel[T], proposalStep: ContinuousDistr[T],
  init: T, burnIn: Long = 0, dropCount: Int = 0)(
  implicit rand:RandBasis=Rand, f: Field[T]) extends
  BaseMetropolisHastings[T](logLikelihoodF.run, init, burnIn, dropCount)(rand) { self =>

  override def proposalDraw(x: T): T = f.plus(proposalStep.draw(), x)

  val proposal = proposalStep

  def observe(x: T) = this.copy(logLikelihoodF, proposal, burnIn = 0L, init = x)

  override def logTransitionProbability(start: T, end: T) = 0.0
}

abstract class HamiltonianMetropolis[T](
  logLikelihoodF: DifferentiableLikelihoodModel[T],
  proposalDistr: ContinuousDistr[T], init: T, burnIn: Long = 0)(
  implicit rand:RandBasis=Rand, f: Field[T]) extends
  GeneralMetropolisHastings[T](
    logLikelihoodF, proposalDistr,
    init, burnIn, dropCount = 0) {

  override def proposalDraw(x: T): T = f.plus(proposalStep.draw(), x)

  override val proposal = proposalDistr

  override def observe(x: T) = this.copy(logLikelihoodF, proposal, burnIn = 0L, init = x)

  override def logTransitionProbability(start: T, end: T) = 0.0


}