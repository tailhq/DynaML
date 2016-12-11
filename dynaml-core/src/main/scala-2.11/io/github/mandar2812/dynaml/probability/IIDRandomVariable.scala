package io.github.mandar2812.dynaml.probability

import breeze.stats.distributions.{ContinuousDistr, Density}
import io.github.mandar2812.dynaml.pipes.DataPipe

/**
  * An independent and identically distributed
  * [[RandomVariable]] represented as a [[Stream]]
  *
  * */
trait IIDRandomVariable[D, R <: RandomVariable[D]] extends RandomVariable[Stream[D]] {

  val baseRandomVariable: R

  val num: Int

  override val sample = DataPipe(() => Stream.tabulate[D](num)(_ => baseRandomVariable.sample()))
}

object IIDRandomVariable {

  def apply[D, R <: RandomVariable[D]](base: R)(n: Int) = new IIDRandomVariable[D, R] {

    val baseRandomVariable = base

    val num = n
  }

  def apply[C, D, R <: RandomVariable[D]](base: DataPipe[C, R])(n: Int) =
    DataPipe((c: C) => new IIDRandomVariable[D, R] {

      val baseRandomVariable = base(c)

      val num = n

    })

}

/**
  * An i.i.d random variable with a defined distribution.
  * */
trait IIDRandomVarDistr[
D, Dist <: Density[D],
R <: RandomVarWithDistr[D, Dist]] extends IIDRandomVariable[D, R]
  with RandomVarWithDistr[Stream[D], Density[Stream[D]]] {

  val baseRandomVariable: R

  val num: Int

  override val sample = DataPipe(() => Stream.tabulate[D](num)(_ => baseRandomVariable.sample()))

  override val underlyingDist = new Density[Stream[D]] {
    override def apply(x: Stream[D]): Double = x.map(baseRandomVariable.underlyingDist(_)).product
  }
}

object IIDRandomVarDistr {
  def apply[D](base : RandomVarWithDistr[D, Density[D]])(n: Int) =
    new IIDRandomVarDistr[D, Density[D], RandomVarWithDistr[D, Density[D]]] {
      val baseRandomVariable = base
      val num = n
    }

  def apply[D](base : ContinuousDistrRV[D])(n: Int) =
    new IIDRandomVarDistr[D, ContinuousDistr[D], ContinuousDistrRV[D]] {
      val baseRandomVariable = base
      val num = n
    }
}


