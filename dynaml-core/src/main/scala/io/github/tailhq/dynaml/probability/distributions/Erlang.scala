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
package io.github.tailhq.dynaml.probability.distributions

import breeze.numerics.{digamma, lgamma}
import breeze.stats.distributions._
import io.github.tailhq.dynaml.utils._

/**
  * <h3><a href="https://en.wikipedia.org/wiki/Erlang_distribution">Erlang Distribution.</a></h3>
  *
  * The Erlang distribution was developed by A. K. Erlang to examine the number of telephone calls which might
  * be made at the same time to the operators of the switching stations. This work on telephone traffic engineering
  * has been expanded to consider waiting times in queueing systems in general. The distribution is now used in the
  * fields of stochastic processes and of biomathematics.
  *
  * */
case class Erlang(shape: Int, rate: Double)(implicit rand: RandBasis = Rand) extends
  ContinuousDistr[Double] with HasCdf with Moments[Double, Double] {

  require(shape > 0 && rate > 0, "Parameters of Erlang distribution must be positive")

  private val g = Gamma(shape, rate)

  override def unnormalizedLogPdf(x: Double) = shape*math.log(rate) + (shape - 1)*math.log(x) - (rate*x)

  override def logNormalizer = math.log(factorial(shape-1))

  override def probability(x: Double, y: Double) = cdf(y) - cdf(x)

  override def cdf(x: Double) = 1d - (0 until shape).map(i => math.exp(-rate*x)*math.pow(rate*x, i)/factorial(i)).sum

  override def mean = shape/rate

  override def variance = shape/(rate*rate)

  override def entropy = (1-shape)*digamma(shape) + lgamma(shape) - math.log(rate) + shape

  override def mode = (shape-1)/rate

  override def draw() = g.draw()
}
