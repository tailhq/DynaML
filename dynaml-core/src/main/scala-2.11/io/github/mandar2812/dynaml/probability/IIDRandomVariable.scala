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

import breeze.stats.distributions.{ContinuousDistr, Density, DiscreteDistr, Rand}
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.probability.distributions.{AbstractContinuousDistr, AbstractDiscreteDistr, GenericDistribution}

/**
  * An independent and identically distributed
  * [[RandomVariable]] represented as a [[Stream]]
  *
  * @tparam D The base domain
  * @tparam R Random variable defined over the base domain
  * */
trait IIDRandomVariable[D, +R <: RandomVariable[D]] extends RandomVariable[Stream[D]] {

  val baseRandomVariable: R

  val num: Int

  override val sample = DataPipe(() => Stream.tabulate[D](num)(_ => baseRandomVariable.draw))
}

object IIDRandomVariable {

  def apply[D](base: RandomVariable[D])(n: Int) = new IIDRandomVariable[D, RandomVariable[D]] {

    val baseRandomVariable = base

    val num = n
  }

  def apply[C, D](base: DataPipe[C, RandomVariable[D]])(n: Int) =
    DataPipe((c: C) => new IIDRandomVariable[D, RandomVariable[D]] {

      val baseRandomVariable = base(c)

      val num = n

    })

}

/**
  * An i.i.d random variable with a defined distribution.
  *
  * @tparam D Base domain
  * @tparam Dist A breeze distribution defined over the
  *              base domain.
  * @tparam R A random variable defined over the base domain
  *           and having distribution of type [[Dist]]
  * */
trait IIDRandomVarDistr[
D, +Dist <: Density[D] with Rand[D],
+R <: RandomVarWithDistr[D, Dist]]
  extends IIDRandomVariable[D, R] with
    RandomVarWithDistr[Stream[D], Density[Stream[D]] with Rand[Stream[D]]] {

  val baseRandomVariable: R

  val num: Int

  override val sample = DataPipe(() => Stream.tabulate[D](num)(_ => baseRandomVariable.draw))

  override val underlyingDist = new GenericDistribution[Stream[D]] {
    override def apply(x: Stream[D]): Double = x.map(baseRandomVariable.underlyingDist(_)).product

    override def draw() = Stream.tabulate[D](num)(_ => baseRandomVariable.underlyingDist.draw())
  }
}

object IIDRandomVarDistr {
  def apply[D, Dist <: Density[D] with Rand[D]](base : RandomVarWithDistr[D, Dist])(n: Int)
  : IIDRandomVarDistr[D, Dist, RandomVarWithDistr[D, Dist]] =
    new IIDRandomVarDistr[D, Dist, RandomVarWithDistr[D, Dist]] {
      val baseRandomVariable = base
      val num = n
    }

}

/**
  * An IID Random variable constructed
  * from a continuous random variable having
  * a defined distribution.
  *
  * @tparam D Base domain
  * @tparam Distr A breeze probability density defined over the
  *              base domain.
  * @tparam R A random variable defined over the base domain
  *           and having distribution of type [[Distr]]
  * */
trait IIDContinuousRVDistr[
D, +Distr <: ContinuousDistr[D],
+R <: ContinuousRVWithDistr[D, Distr]] extends
  IIDRandomVarDistr[D, Distr, R] {

  override val underlyingDist = new AbstractContinuousDistr[Stream[D]] {

    override def unnormalizedLogPdf(x: Stream[D]) = x.map(baseRandomVariable.underlyingDist.unnormalizedLogPdf).sum

    override def logNormalizer = num*baseRandomVariable.underlyingDist.logNormalizer

    override def draw() = Stream.tabulate[D](num)(_ => baseRandomVariable.underlyingDist.draw())
  }

}

object IIDContinuousRVDistr {

  def apply[
  D, Distr <: ContinuousDistr[D],
  R <: ContinuousRVWithDistr[D, Distr]](base: R)(n: Int): IIDContinuousRVDistr[D, Distr, R] =
    new IIDContinuousRVDistr[D, Distr, R] {
      override val baseRandomVariable = base
      override val num = n
    }


}

trait IIDDiscreteRVDistr[
D, +Distr <: DiscreteDistr[D],
+R <: DiscreteRVWithDistr[D, Distr]] extends
  IIDRandomVarDistr[D, Distr, R] {

  override val underlyingDist = new AbstractDiscreteDistr[Stream[D]] {
    override def probabilityOf(x: Stream[D]) = x.map(baseRandomVariable.underlyingDist.probabilityOf).product

    override def draw() = Stream.tabulate[D](num)(_ => baseRandomVariable.underlyingDist.draw())
  }
}

object IIDDiscreteRVDistr {
  def apply[
  D, Distr <: DiscreteDistr[D],
  R <: DiscreteRVWithDistr[D, Distr]](base: R)(n: Int): IIDDiscreteRVDistr[D, Distr, R] =
    new IIDDiscreteRVDistr[D, Distr, R] {
      override val baseRandomVariable = base
      override val num = n
    }

}


