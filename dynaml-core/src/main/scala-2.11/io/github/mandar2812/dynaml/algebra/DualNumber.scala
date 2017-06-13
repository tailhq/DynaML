package io.github.mandar2812.dynaml.algebra

import spire.algebra.Field

/**
  * Represents a dual number over an abstract field
  *
  * z = a + b &epsilon;
  *
  * @tparam I Base Field
  * @param a The 'real' component
  * @param b The auxiliary component
  * @author mandar2812 date 13/06/2017.
  * */
case class DualNumber[I](a: I, b:I)(implicit f: Field[I]) {
  self =>

  /**
    * Dual number addition
    * */
  def +(other: DualNumber[I]): DualNumber[I] = other match {
    case _: DualZero[I] => self
    case _ => DualNumber[I](
      f.plus(self.a, other.a),
      f.plus(self.b, other.b))
  }

  def +(other: I): DualNumber[I] = DualNumber[I](f.plus(self.a, other), self.b)

  /**
    * Dual number subtraction
    * */
  def -(other: DualNumber[I]): DualNumber[I] = other match {
    case _: DualZero[I] => self
    case _ => DualNumber[I](
      f.minus(self.a, other.a),
      f.minus(self.b, other.b))
  }

  def -(other: I): DualNumber[I] = DualNumber[I](f.minus(self.a, other), self.b)

  /**
    * Dual number multiplication
    * */
  def *(other: DualNumber[I]): DualNumber[I] = other match {
    case _: DualZero[I] => DualNumber.zero[I]
    case _ => DualNumber[I](
      f.times(self.a, other.a),
      f.plus(
        f.times(self.a, other.b),
        f.times(self.b, other.a)
      )
    )
  }

  def *(other: I): DualNumber[I] = DualNumber[I](
    f.times(self.a, other),
    f.times(self.b, other))

}

/**
  * Dual number zero
  * */
sealed class DualZero[I](implicit f: Field[I]) extends DualNumber(f.zero, f.zero)

object DualNumber {

  /**
    * Create the zero dual number instance for the type [[I]]
    * */
  def zero[I](implicit f: Field[I]) = new DualZero[I]

}