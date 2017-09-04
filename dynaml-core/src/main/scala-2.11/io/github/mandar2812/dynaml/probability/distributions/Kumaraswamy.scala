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
package io.github.mandar2812.dynaml.probability.distributions

import breeze.numerics._
import breeze.stats.distributions._
import io.github.mandar2812.dynaml.utils._

/**
  * <h3><a href="https://en.wikipedia.org/wiki/Kumaraswamy_distribution">Kumaraswamy Distribution</a><h3>
  *
  * A probability distribution for a univariate random variable
  * which takes values in the interval [0, 1]. The Kumaraswamy distribution
  * is closely related to the Beta distribution.
  *
  * */
case class Kumaraswamy(a: Double, b: Double)(implicit rand: RandBasis = Rand) extends
    ContinuousDistr[Double] with HasCdf with Moments[Double, Double] {

  val (loga, logb) = (log(a), log(b))

  private val m = (n: Int) => math.exp(lbeta(1d + n/a, b) + math.log(b))

  override def unnormalizedLogPdf(x: Double) = (a-1)*log(x) + (b-1)*log(1 - pow(x, a))

  override def logNormalizer = - loga - logb

  override def probability(x: Double, y: Double) = cdf(y) - cdf(x)

  override def cdf(x: Double) = 1d - math.pow(1d - math.pow(x, a), b)

  override def mean = math.exp(lgamma(1 + 1/a) + lgamma(b) + math.log(b) - lgamma(1 + (1/a) + b))

  override def variance = m(2) - m(1)*m(1)

  override def entropy = (1d - (1d/a)) + (1d - (1d/b))*H(b) + math.log(a*b)

  override def mode =
    if(a >= 1 && b>= 1 && a != b) math.pow((a - 1)/(a*b - 1), 1/a)
    else Double.NaN

  override def draw() = math.pow(1 - math.pow(1 - rand.uniform.draw(), 1/b), 1/a)
}
