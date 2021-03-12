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
package io.github.tailhq.dynaml.probability

import io.github.tailhq.dynaml.pipes.DataPipe
import spire.implicits.cfor

/**
  * @author tailhq date 09/01/2017.
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
        var candidate = prior.draw
        var generatedData = likelihood(candidate).draw

        cfor(1)(count =>
          count < MAX_ITERATIONS && !acceptance(data, generatedData),
          count => count + 1)(count => {
          candidate = prior.draw
          generatedData = likelihood(candidate).draw
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