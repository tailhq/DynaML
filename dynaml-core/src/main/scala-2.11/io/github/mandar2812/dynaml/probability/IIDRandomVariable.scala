package io.github.mandar2812.dynaml.probability

import breeze.stats.distributions.{ContinuousDistr, Density, Rand}
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.probability.distributions.{AbstractContinuousDistr, GenericDistribution}

/**
  * An independent and identically distributed
  * [[RandomVariable]] represented as a [[Stream]]
  *
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

/**
  * An IID Random variable constructed
  * from a continuous random variable having
  * a defined distribution.
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

object IIDRandomVarDistr {
  def apply[D, Dist <: Density[D] with Rand[D]](base : RandomVarWithDistr[D, Dist])(n: Int) =
    new IIDRandomVarDistr[D, Dist, RandomVarWithDistr[D, Dist]] {
      val baseRandomVariable = base
      val num = n
    }

  def apply[
  D, Distr <: ContinuousDistr[D],
  R <: ContinuousRVWithDistr[D, Distr]](base: R)(n: Int) = new IIDContinuousRVDistr[D, Distr, R] {
    override val baseRandomVariable = base
    override val num = n
  }

}


