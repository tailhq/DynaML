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

import spire.algebra.{Field, VectorSpace}
import breeze.linalg.DenseVector
import breeze.stats.distributions.{ContinuousDistr, Moments}
import io.github.tailhq.dynaml.algebra.{PartitionedPSDMatrix, PartitionedVector}
import io.github.tailhq.dynaml.analysis.PartitionedVectorField
import io.github.tailhq.dynaml.pipes.DataPipe
import io.github.tailhq.dynaml.probability.distributions.{
BlockedMultiVariateGaussian, HasErrorBars,
MixtureDistribution, MixtureWithConfBars}


/**
  * Top level class for representing
  * random variable mixtures.
  *
  * @tparam Domain Domain over which each mixture
  *                component is defined
  *
  * @tparam BaseRV The type of each mixture component, must
  *                be a sub type of [[RandomVariable]]
  * @author tailhq date 14/06/2017.
  * */
trait MixtureRV[Domain, BaseRV <: RandomVariable[Domain]] extends
  RandomVariable[Domain] {

  val mixture_selector: MultinomialRV

  val components: Seq[BaseRV]

}

object MixtureRV {

  def apply[Domain, BaseRV <: RandomVariable[Domain]](
    distributions: Seq[BaseRV],
    weights: DenseVector[Double]): MixtureRV[Domain, BaseRV] =
    new MixtureRV[Domain, BaseRV] {

      override val mixture_selector = MultinomialRV(weights)

      override val components = distributions

      val sample = mixture_selector.sample > DataPipe((i: Int) => components(i).draw)
    }

}

/**
  * A random variable mixture over a continuous domain
  *
  * @tparam Domain Domain over which each mixture
  *                component is defined
  *
  * @tparam BaseRV The type of each mixture component, must
  *                be a sub type of [[ContinuousRandomVariable]]
  *
  * @author tailhq date 14/06/2017
  * */
trait ContinuousMixtureRV[Domain, BaseRV <: ContinuousRandomVariable[Domain]] extends
  ContinuousRandomVariable[Domain] with
  MixtureRV[Domain, BaseRV]

/**
  * A random variable mixture over a continuous domain,
  * having a computable probability distribution
  *
  * @tparam Domain Domain over which each mixture
  *                component is defined
  *
  * @tparam BaseRV The type of each mixture component, must
  *                be a sub type of [[ContinuousRVWithDistr]]
  *
  * @author tailhq date 14/06/2017
  * */
class ContinuousDistrMixture[Domain, BaseRV <: ContinuousRVWithDistr[Domain, ContinuousDistr[Domain]]](
  distributions: Seq[BaseRV],
  selector: MultinomialRV) extends
  ContinuousMixtureRV[Domain, BaseRV] with
  ContinuousDistrRV[Domain] {

  override val mixture_selector = selector

  override val components = distributions

  override val underlyingDist = MixtureDistribution(
    components.map(_.underlyingDist),
    selector.underlyingDist.params)
}

/**
  * A random variable mixture over a continuous domain,
  * having a computable probability distribution and ability
  * to generate error bars/confidence intervals.
  *
  * @tparam Domain Domain over which each mixture
  *                component is defined
  *
  * @tparam Var The type of second moment i.e. variance of the random variables
  *             defined over [[Domain]].
  *
  * @tparam BaseDistr The type of each mixture component distribution,
  *                   must be a continuous breeze distribution with
  *                   defined moments and implement [[HasErrorBars]]
  *
  * @author tailhq date 14/06/2017
  * */
private[dynaml] class ContMixtureRVBars[Domain, Var,
BaseDistr <: ContinuousDistr[Domain] with Moments[Domain, Var] with HasErrorBars[Domain]](
  distributions: Seq[BaseDistr],
  selector: MultinomialRV)(
  implicit v: VectorSpace[Domain, Double]) extends
  ContinuousDistrMixture[Domain, ContinuousRVWithDistr[Domain, ContinuousDistr[Domain]]](
    distributions.map(RandomVariable(_)), selector) {

  override val underlyingDist = MixtureWithConfBars(distributions, selector.underlyingDist.params)
}

/**
  * Contains convenience methods for creating
  * various mixture random variable instances.
  * */
object ContinuousDistrMixture {

  def apply[Domain, BaseRV <: ContinuousDistrRV[Domain]](
    distributions: Seq[BaseRV],
    selector: DenseVector[Double]): ContinuousDistrMixture[Domain, BaseRV] =
    new ContinuousDistrMixture(distributions, MultinomialRV(selector))

  def apply[Domain, Var, BaseDistr <: ContinuousDistr[Domain] with Moments[Domain, Var] with HasErrorBars[Domain]](
    distributions: Seq[BaseDistr], selector: DenseVector[Double])(
    implicit v: VectorSpace[Domain, Double]) =
    new ContMixtureRVBars[Domain, Var, BaseDistr](distributions, MultinomialRV(selector))

  def apply(num_elements_per_block: Int)(
    distributions: Seq[BlockedMultiVariateGaussian],
    selector: DenseVector[Double]) = {

    val num_dim = distributions.head.mean.rows

    implicit val ev = PartitionedVectorField(num_dim, num_elements_per_block)

    apply[PartitionedVector, PartitionedPSDMatrix, BlockedMultiVariateGaussian](distributions, selector)
  }

}
