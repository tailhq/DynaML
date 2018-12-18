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

import breeze.stats.distributions.{ContinuousDistr, Density, Rand}
import io.github.mandar2812.dynaml.pipes.DataPipe

/**
  * @author mandar2812 date: 07/01/2017.
  * A probability distribution driven by a set
  * of parameters.
  */
trait LikelihoodModel[Q] extends DataPipe[Q, Double]{


  /**
    * Returns log p(q)
    * */
  def loglikelihood(q: Q): Double

  override def run(data: Q) = loglikelihood(data)

}

object LikelihoodModel {
  def apply[Q, T <: Density[Q] with Rand[Q]](d: T) = new LikelihoodModel[Q] {
    val density = d

    /**
      * Returns log p(q)
      **/
    override def loglikelihood(q: Q) = density.logApply(q)

  }

  def apply[Q](d: ContinuousDistr[Q]) = new LikelihoodModel[Q] {
    val density = d

    /**
      * Returns log p(q)
      **/
    override def loglikelihood(q: Q) = density.logPdf(q)
  }

  def apply[Q](f: (Q) => Double) = new LikelihoodModel[Q] {
    /**
      * Returns log p(q)
      **/
    override def loglikelihood(q: Q) = f(q)
  }

}

trait DifferentiableLikelihoodModel[Q] extends LikelihoodModel[Q] {
  /**
    * Returns d(log p(q|r))/dr
    * */
  def gradLogLikelihood(q: Q): Q
}

object DifferentiableLikelihoodModel {

  def apply[Q](d: ContinuousDistr[Q], grad: DataPipe[Q, Q]): DifferentiableLikelihoodModel[Q] =
    new DifferentiableLikelihoodModel[Q] {

      val density = d

      val gradPotential = grad

      /**
        * Returns log p(q)
        **/
      override def loglikelihood(q: Q) = density.logPdf(q)

      /**
        * Returns d(log p(q|r))/dr
        **/
      override def gradLogLikelihood(q: Q) = gradPotential(q)
    }

  def apply[Q](f: (Q) => Double, grad: (Q) => Q) = new DifferentiableLikelihoodModel[Q] {
    /**
      * Returns log p(q)
      **/
    override def loglikelihood(q: Q) = f(q)

    /**
      * Returns d(log p(q|r))/dr
      **/
    override def gradLogLikelihood(q: Q) = grad(q)
  }
}