package io.github.mandar2812.dynaml.probability

import breeze.stats.distributions.{ContinuousDistr, Density, DiscreteDistr}
import io.github.mandar2812.dynaml.pipes.{BifurcationPipe, DataPipe}
import spire.algebra.{Field, NRoot}


/**
  * @author mandar date: 26/7/16.
  *
  * Represents a bare bones representation
  * of a random variable only in terms of
  * samples.
  */
trait RandomVariable[Domain] {

  /**
    * Generate a sample from
    * the random variable
    *
    * */
  val sample: DataPipe[Unit, Domain]

  def :*:[Domain1](other: RandomVariable[Domain1]): RandomVariable[(Domain, Domain1)] = {
    val sam = this.sample
    RandomVariable(BifurcationPipe(sam,other.sample))
  }
}


trait HasDistribution[Domain] {
  val underlyingDist: Density[Domain]
}

trait RandomVarDistribution[Domain] extends RandomVariable[Domain] with HasDistribution[Domain] {
  override val underlyingDist: Density[Domain]

}

abstract class ContinuousRandomVariable[Domain](implicit ev: Field[Domain]) extends RandomVariable[Domain] {

  def +(other: ContinuousRandomVariable[Domain]): ContinuousRandomVariable[Domain] = {
    val sam = this.sample.run _

    new ContinuousRandomVariable[Domain] {
    override val sample = DataPipe(() => other.sample()) >
        DataPipe((first: Domain) => ev.plus(first, sam()))
    }
  }

  def -(other: ContinuousRandomVariable[Domain]): ContinuousRandomVariable[Domain] = {
    val sam = this.sample.run _

    new ContinuousRandomVariable[Domain] {
      override val sample = DataPipe(() => other.sample()) >
        DataPipe((first: Domain) => ev.minus(sam(), first))
    }
  }

  def *(other: ContinuousRandomVariable[Domain]): ContinuousRandomVariable[Domain] = {
    val sam = this.sample.run _

    new ContinuousRandomVariable[Domain] {
      override val sample = DataPipe(() => other.sample()) >
        DataPipe((first: Domain) => ev.times(first, sam()))
    }
  }

  def +(other: Domain): ContinuousRandomVariable[Domain] = {
    val sam = this.sample.run _

    new ContinuousRandomVariable[Domain] {
      override val sample = DataPipe(() => ev.plus(other, sam()))
    }
  }

  def -(other: Domain): ContinuousRandomVariable[Domain] = {
    val sam = this.sample.run _

    new ContinuousRandomVariable[Domain] {
      override val sample = DataPipe(() => ev.minus(sam(), other))
    }
  }

  def *(other: Domain): ContinuousRandomVariable[Domain] = {
    val sam = this.sample.run _

    new ContinuousRandomVariable[Domain] {
      override val sample = DataPipe(() => ev.times(other, sam()))
    }
  }
}

trait ContinuousDistrRV[Domain]
  extends ContinuousRandomVariable[Domain]
  with HasDistribution[Domain] {

  override val underlyingDist: ContinuousDistr[Domain]

  override val sample = DataPipe(() => underlyingDist.sample())
}

trait DiscreteDistrRV[Domain] extends RandomVarDistribution[Domain] {

  override val underlyingDist: DiscreteDistr[Domain]

  override val sample = DataPipe(() => underlyingDist.sample())
}


object RandomVariable {

  def apply[O](f: () => O) = new RandomVariable[O] {
    val sample = DataPipe(f)
  }

  def apply[O](f: DataPipe[Unit, O]) = new RandomVariable[O] {
    val sample = f
  }

  def apply[O](d: ContinuousDistr[O])(implicit ev: Field[O]) = new ContinuousDistrRV[O] {
    override val underlyingDist = d
  }

  def apply[O](d: DiscreteDistr[O]) = new DiscreteDistrRV[O] {
    override val underlyingDist = d
  }
}
