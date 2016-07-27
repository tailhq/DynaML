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
  *
  * @tparam Domain The domain or space over which
  *                the random variable is defined;
  *                i.e. the range of values it can take
  */
abstract class RandomVariable[Domain] {

  /**
    * Generate a sample from
    * the random variable
    *
    * */
  val sample: DataPipe[Unit, Domain]

  /**
    * Outputs the cartesian product between
    * two random variables.
    *
    * @tparam Domain1 The domain of the other random variable
    *
    * @param other The random variable which forms the second
    *              component of the cartesian product.
    *
    * */
  def :*:[Domain1](other: RandomVariable[Domain1]): RandomVariable[(Domain, Domain1)] = {
    val sam = this.sample
    RandomVariable(BifurcationPipe(sam,other.sample))
  }
}

/**
  * This trait is used to denote
  * random variables which have an
  * underlying probability density function.
  *
  * */
trait HasDistribution[Domain] {

  /**
    * The actual probability density
    * function is represented as a breeze
    * [[Density]] object.
    * */
  val underlyingDist: Density[Domain]
}

/**
  * A (univariate) continuous random variable that has an associated
  * probability density function.
  *
  * @tparam Domain The domain over which the random variable takes values.
  * @param ev An implicit parameter, which represents a [[Field]] defined over
  *           the set [[Domain]].
  * */
abstract class ContinuousRandomVariable[Domain](implicit ev: Field[Domain]) extends RandomVariable[Domain] {

  /**
    * Return the random variable that results from adding a provided
    * random variable to the current random variable.
    *
    * @param other The random variable to be added to this.
    * */
  def +(other: ContinuousRandomVariable[Domain]): ContinuousRandomVariable[Domain] = {
    val sam = this.sample.run _

    new ContinuousRandomVariable[Domain] {
    override val sample = DataPipe(() => other.sample()) >
        DataPipe((first: Domain) => ev.plus(first, sam()))
    }
  }

  /**
    * Return the random variable that results from subtracting a provided
    * random variable to the current random variable.
    *
    * @param other The random variable to be subtracted from this.
    * */
  def -(other: ContinuousRandomVariable[Domain]): ContinuousRandomVariable[Domain] = {
    val sam = this.sample.run _

    new ContinuousRandomVariable[Domain] {
      override val sample = DataPipe(() => other.sample()) >
        DataPipe((first: Domain) => ev.minus(sam(), first))
    }
  }

  /**
    * Return the random variable that results from multiplying a provided
    * random variable to the current random variable.
    *
    * @param other The random variable to be multiplied to this.
    * */
  def *(other: ContinuousRandomVariable[Domain]): ContinuousRandomVariable[Domain] = {
    val sam = this.sample.run _

    new ContinuousRandomVariable[Domain] {
      override val sample = DataPipe(() => other.sample()) >
        DataPipe((first: Domain) => ev.times(first, sam()))
    }
  }

  /**
    * Return the random variable that results from adding a provided
    * quantity in the [[Domain]] to the current random variable.
    *
    * @param other The value to be added to this random variable.
    * */
  def +(other: Domain): ContinuousRandomVariable[Domain] = {
    val sam = this.sample.run _

    new ContinuousRandomVariable[Domain] {
      override val sample = DataPipe(() => ev.plus(other, sam()))
    }
  }


  /**
    * Return the random variable that results from subtracting a provided
    * quantity in the [[Domain]] from the current random variable.
    *
    * @param other The value to be subtracted from this random variable.
    * */
  def -(other: Domain): ContinuousRandomVariable[Domain] = {
    val sam = this.sample.run _

    new ContinuousRandomVariable[Domain] {
      override val sample = DataPipe(() => ev.minus(sam(), other))
    }
  }

  /**
    * Return the random variable that results from multiplying a provided
    * quantity in the [[Domain]] to the current random variable.
    *
    * @param other The value to be multiplied to this random variable.
    * */
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

trait DiscreteDistrRV[Domain]
  extends RandomVariable[Domain]
    with HasDistribution[Domain] {

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

/**
  * An independent and identically distributed
  * [[RandomVariable]] represented as a [[Stream]]
  *
  * */
trait IIDRandomVariable[D, R[D] <: RandomVariable[D]] extends RandomVariable[Stream[D]] {

  val baseRandomVariable: R[D]

  val num: Int

  override val sample = DataPipe(() => Stream.tabulate[D](num)(_ => baseRandomVariable.sample()))
}

object IIDRandomVariable {

  def apply[D, R[D] <: RandomVariable[D]](base: R[D])(n: Int) = new IIDRandomVariable[D, R] {

    val baseRandomVariable = base

    val num = n
  }
}