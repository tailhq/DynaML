package io.github.mandar2812.dynaml.probability

import breeze.stats.distributions.{ContinuousDistr, Density, DiscreteDistr}
import io.github.mandar2812.dynaml.pipes.{BifurcationPipe, DataPipe}
import spire.algebra.Field
import io.github.mandar2812.dynaml.utils._


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

  self =>
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
  def :*[Domain1](other: RandomVariable[Domain1]): RandomVariable[(Domain, Domain1)] = {
    val sam = this.sample
    RandomVariable(BifurcationPipe(sam, other.sample))
  }

  /**
    * Create an iid random variable from the current (this)
    * @param n The number of iid samples of the base random variable.
    * */
  def iid(n: Int): IIDRandomVariable[Domain, RandomVariable[Domain]] =
  new IIDRandomVariable[Domain, RandomVariable[Domain]] {
    override val baseRandomVariable: RandomVariable[Domain] = self
    override val num: Int = n
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
  * Random variable defined over a continuous domain.
  *
  * @tparam Domain The domain over which the random variable takes values.
  * */
trait ContinuousRandomVariable[Domain] extends RandomVariable[Domain] {

  /**
    * Return the random variable that results from adding a provided
    * random variable to the current random variable.
    *
    * @param other The random variable to be added to this.
    * @param ev An implicit parameter, which represents a [[Field]] defined over
    *           the set [[Domain]].
    * */
  def +(other: ContinuousRandomVariable[Domain])(implicit ev: Field[Domain]): ContinuousRandomVariable[Domain] = {
    val sam = this.sample.run _

    new ContinuousRandomVariable[Domain] {
    override val sample = DataPipe(() => other.sample()) >
        DataPipe((first: Domain) => ev.plus(first, sam()))
    }
  }

  /**
    * Return the random variable that results from subtracting a provided
    * random variable from the current random variable.
    *
    * @param other The random variable to be subtracted from this.
    * @param ev An implicit parameter, which represents a [[Field]] defined over
    *           the set [[Domain]].
    * */
  def -(other: ContinuousRandomVariable[Domain])(implicit ev: Field[Domain]): ContinuousRandomVariable[Domain] = {
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
    * @param ev An implicit parameter, which represents a [[Field]] defined over
    *           the set [[Domain]].
    * */
  def *(other: ContinuousRandomVariable[Domain])(implicit ev: Field[Domain]): ContinuousRandomVariable[Domain] = {
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
    * @param ev An implicit parameter, which represents a [[Field]] defined over
    *           the set [[Domain]].
    * */
  def +(other: Domain)(implicit ev: Field[Domain]): ContinuousRandomVariable[Domain] = {
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
    * @param ev An implicit parameter, which represents a [[Field]] defined over
    *           the set [[Domain]].
    * */
  def -(other: Domain)(implicit ev: Field[Domain]): ContinuousRandomVariable[Domain] = {
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
    * @param ev An implicit parameter, which represents a [[Field]] defined over
    *           the set [[Domain]].
    * */
  def *(other: Domain)(implicit ev: Field[Domain]): ContinuousRandomVariable[Domain] = {
    val sam = this.sample.run _

    new ContinuousRandomVariable[Domain] {
      override val sample = DataPipe(() => ev.times(other, sam()))
    }
  }
}

/**
  * A random variable which has a computable probability
  * density.
  *
  * @tparam Domain The set over which the random variable takes its values.
  * @tparam Dist A breeze probability density defined on the
  *              [[Domain]]
  * */
trait RandomVarWithDistr[Domain, Dist <: Density[Domain]]
  extends RandomVariable[Domain]
    with HasDistribution[Domain] {

  self =>

  override val underlyingDist: Dist

  /**
    * Cartesian product with another random variables which
    * has a defined probability distribution.
    *
    * */
  def :*[Domain1, Dist1 <: Density[Domain1]](other: RandomVarWithDistr[Domain1, Dist1]):
  RandomVarWithDistr[(Domain, Domain1), Density[(Domain, Domain1)]] =
  new RandomVarWithDistr[(Domain, Domain1), Density[(Domain, Domain1)]] {

    val underlyingDist: Density[(Domain, Domain1)] = new Density[(Domain, Domain1)] {
      override def apply(x: (Domain, Domain1)): Double = self.underlyingDist(x._1)*other.underlyingDist(x._2)
    }

    val sample = BifurcationPipe(self.sample, other.sample)
  }

  //TODO: Figure out implementation of following iid method
  /*override def iid(n: Int): IIDRandomVariable[Domain, RandomVarWithDistr[Domain, Dist]] =
    new IIDRandomVarDistr[Domain, Dist, RandomVarWithDistr[Domain, Dist]] {
      override val baseRandomVariable: RandomVarWithDistr[Domain, Dist] = self
      override val num: Int = n
    }*/

}

/**
  * A continuous random variable that has an associated
  * probability density function.
  *
  * */
abstract class ContinuousDistrRV[Domain]
  extends ContinuousRandomVariable[Domain]
    with RandomVarWithDistr[Domain, ContinuousDistr[Domain]] {

  override val underlyingDist: ContinuousDistr[Domain]

  override val sample = DataPipe(() => underlyingDist.sample())

  def :*[Domain1](other: ContinuousDistrRV[Domain1])(implicit ev: Field[Domain], ev1: Field[Domain1])
  : ContinuousDistrRV[(Domain, Domain1)] = {
    val sam = this.underlyingDist.draw _
    val dist = this.underlyingDist
    val sampleFirst = this.sample

    implicit val ev3: Field[(Domain, Domain1)] = productField(ev, ev1)

    new ContinuousDistrRV[(Domain, Domain1)] {

      val underlyingDist: ContinuousDistr[(Domain, Domain1)] = new ContinuousDistr[(Domain, Domain1)] {
        override def unnormalizedLogPdf(x: (Domain, Domain1)): Double =
          dist.unnormalizedLogPdf(x._1) +
          other.underlyingDist.unnormalizedLogPdf(x._2)

        override def logNormalizer: Double = dist.logNormalizer + other.underlyingDist.logNormalizer

        override def draw(): (Domain, Domain1) = (sam(), other.underlyingDist.draw())
      }

      override val sample = BifurcationPipe(sampleFirst, other.sample)
    }

  }


}

/**
  * A random variable which takes discrete or categorical values.
  * */
abstract class DiscreteDistrRV[Domain]
  extends RandomVarWithDistr[Domain, DiscreteDistr[Domain]] {

  override val underlyingDist: DiscreteDistr[Domain]

  override val sample = DataPipe(() => underlyingDist.sample())

  /**
    * Cartesian product with another discrete valued random variable.
    *
    * */
  def :*[Domain1](other: DiscreteDistrRV[Domain1]): DiscreteDistrRV[(Domain, Domain1)] = {

    val dist = this.underlyingDist
    val sam = this.underlyingDist.draw _

    new DiscreteDistrRV[(Domain, Domain1)] {
      override val underlyingDist: DiscreteDistr[(Domain, Domain1)] = new DiscreteDistr[(Domain, Domain1)] {

        override def probabilityOf(x: (Domain, Domain1)): Double =
          dist.probabilityOf(x._1)*other.underlyingDist.probabilityOf(x._2)

        override def draw(): (Domain, Domain1) = (sam(), other.underlyingDist.draw())
      }
    }

  }

}

/**
  * Companion object of the [[RandomVariable]] class.
  * Contains apply methods which can create random variables.
  * */
object RandomVariable {

  /**
    * @param f A function which generates samples from the random variable
    *          to be created.
    * */
  def apply[O](f: () => O) = new RandomVariable[O] {
    val sample = DataPipe(f)
  }

  def apply[O](f: DataPipe[Unit, O]) = new RandomVariable[O] {
    val sample = f
  }

  /**
    * Creates a continuous random variable with a specified distribution.
    * @tparam O The domain of values over which the random variable is defined.
    * @param d A breeze continuous distribution
    * @param ev A spire [[Field]] implementation for the domain [[O]]
    * @return A continuous random variable instance.
    * */
  def apply[O](d: ContinuousDistr[O])(implicit ev: Field[O]) = new ContinuousDistrRV[O] {
    override val underlyingDist = d
  }

  /**
    * Creates a continuous random variable with a specified distribution.
    * @tparam O The domain of values over which the random variable is defined.
    * @param d A breeze discrete distribution
    * @return A discrete random variable instance.
    * */
  def apply[O](d: DiscreteDistr[O]) = new DiscreteDistrRV[O] {
    override val underlyingDist = d
  }
}
