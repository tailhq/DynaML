package io.github.mandar2812.dynaml.analysis

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.algebra.PartitionedVector
import spire.algebra.{Eq, Field, InnerProductSpace, NRoot}
import spire.implicits._

/**
  * @author mandar2812 date: 21/02/2017.
  */
object implicits {

  implicit object innerProdDouble extends InnerProductSpace[Double, Double] {

    override def dot(v: Double, w: Double) = v*w

    override implicit def scalar = Field[Double]

    override def timesl(r: Double, v: Double) = r*v

    override def negate(x: Double) = -x

    override def zero = 0.0

    override def plus(x: Double, y: Double) = x + y
  }

  implicit object innerProdFloat extends InnerProductSpace[Float, Double] {

    override def dot(v: Float, w: Float) = v*w.toDouble

    override implicit def scalar = Field[Double]

    override def timesl(r: Double, v: Float) = r.toFloat*v

    override def negate(x: Float) = -x

    override def zero = 0f

    override def plus(x: Float, y: Float) = x + y
  }

  implicit object innerProdTuple2 extends InnerProductSpace[(Double, Double), Double] {
    
    override def dot(v: (Double, Double), w: (Double, Double)): Double = v._1*w._1 + v._2*w._2

    override implicit def scalar: Field[Double] = Field[Double]

    override def timesl(r: Double, v: (Double, Double)): (Double, Double) = (r*v._1, r*v._2)

    override def negate(x: (Double, Double)): (Double, Double) = (-x._1, -x._2)

    override def zero: (Double, Double) = (0d, 0d)

    override def plus(x: (Double, Double), y: (Double, Double)): (Double, Double) = (x._1+y._1, x._2+y._2)
  }

  implicit object fieldTuple2 extends Field[(Double, Double)] {
    override def quot(a: (Double, Double), b: (Double, Double)): (Double, Double) = (a._1/b._1, a._2/b._2)

    override def mod(a: (Double, Double), b: (Double, Double)): (Double, Double) = (a._1%b._1, a._2%b._2)

    override def gcd(a: (Double, Double), b: (Double, Double))(implicit ev: Eq[(Double, Double)]): (Double, Double) =
      (a._1%b._1, a._2%b._2)

    override def lcm(a: (Double, Double), b: (Double, Double))(implicit ev: Eq[(Double, Double)]): (Double, Double) =
      (a._1%b._1, a._2%b._2)

    override def div(x: (Double, Double), y: (Double, Double)): (Double, Double) = (x._1/y._1, x._2/y._2)

    override def times(x: (Double, Double), y: (Double, Double)): (Double, Double) = (x._1*y._1, x._2*y._2)

    override def negate(x: (Double, Double)): (Double, Double) = (-x._1, -x._2)

    override def zero: (Double, Double) = (0d, 0d)

    override def plus(x: (Double, Double), y: (Double, Double)): (Double, Double) = (x._1+y._1, x._2+y._2)

    override def one: (Double, Double) = (1d, 1d)
  }

  implicit object eqVector extends Eq[DenseVector[Double]] {
    override def eqv(x: DenseVector[Double], y: DenseVector[Double]): Boolean =
      x.toArray.zip(y.toArray).forall(p => p._1 == p._2)
  }

  implicit object eqPartitinedVector extends Eq[PartitionedVector] {
    override def eqv(x: PartitionedVector, y: PartitionedVector): Boolean =
      x._data.map(_._2).zip(y._data.map(_._2)).forall(p => eqVector.eqv(p._1, p._2))
  }

}
