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
package io.github.tailhq.dynaml

import breeze.stats.distributions.ContinuousDistr
import io.github.tailhq.dynaml.pipes.DataPipe

import spire.algebra.InnerProductSpace


/**
  * Contains helper functions pertaining to random variables.
  * @author tailhq date 11/01/2017.
  * */
package object probability {


  /**
    * Number of monte carlo samples to generate.
    * */
  var candidates: Int = 10000

  /**
    * Calculate the monte carlo estimate of the
    * first moment/expectation of a random variable.
    *
    * @tparam I The domain of the random variable.
    * @param rv The random variable from which to sample
    * @param f An implicit object representing an inner product on [[I]]
    * */
  def E[@specialized(Double) I](rv: RandomVariable[I])(implicit f: InnerProductSpace[I, Double]): I =
    f.divr(
      rv.iid(candidates)
        .sample()
        .reduce(
          (x, y) => f.plus(x,y)
        ),
      candidates.toDouble)


  def E(rv: RandomVariable[Double]): Double = rv.iid(candidates).sample().sum/candidates.toDouble



  /**
    * Calculate the entropy of a random variable
    * */
  def entropy[I, Distr <: ContinuousDistr[I]](rv: ContinuousRVWithDistr[I, Distr]): Double = {
    val logp_x: RandomVariable[Double] = MeasurableFunction[I, Double, ContinuousRVWithDistr[I, Distr]](
      rv, DataPipe((sample: I) => -1d*rv.underlyingDist.logPdf(sample)))
    E(logp_x)
  }


  /**
    * KL divergence:
    * @param p The base random variable
    * @param q The random variable used to approximate p
    * @return The Kulback Leibler divergence KL(P||Q)
    * */
  def KL[I, Distr <: ContinuousDistr[I]](
    p: ContinuousRVWithDistr[I, Distr])(
    q: ContinuousRVWithDistr[I, Distr]): Double = {


    val log_q_p: RandomVariable[Double] = MeasurableFunction[I, Double, ContinuousRVWithDistr[I, Distr]](
      p, DataPipe((sample: I) => p.underlyingDist.logPdf(sample)-q.underlyingDist.logPdf(sample)))

    E(log_q_p)
  }


  /**
    * Calculate the (monte carlo approximation to) mean, median, mode, lower and upper confidence interval.
    *
    *
    *
    * @param r Continuous random variable in question
    * @param alpha Probability of exclusion, i.e. return 100(1-alpha) % confidence interval.
    *              alpha should be between 0 and 1
    * */
  def OrderStats(r: ContinuousRVWithDistr[Double, ContinuousDistr[Double]], alpha: Double)
  : (Double, Double, Double, Double, Double) = {

    val samples = r.iid(candidates).sample()

    val mean = samples.sum/candidates.toDouble

    val median = utils.median(samples)

    val (lowerbar, higherbar) = (
      utils.quickselect(samples, math.ceil(candidates*alpha*0.5).toInt),
      utils.quickselect(samples, math.ceil(candidates*(1.0 - 0.5*alpha)).toInt))

    val uDist = r.underlyingDist

    val mode = samples.map(s => (uDist.logPdf(s), s)).max._2

    (mean, median, mode, lowerbar, higherbar)
  }
}
