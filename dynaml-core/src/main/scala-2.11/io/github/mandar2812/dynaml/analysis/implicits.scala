package io.github.mandar2812.dynaml.analysis

import spire.algebra.{Field, InnerProductSpace}

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

  implicit object innerProdTuple2 extends InnerProductSpace[(Double, Double), Double] {
    
    override def dot(v: (Double, Double), w: (Double, Double)): Double = v._1*w._1 + v._2*w._2

    override implicit def scalar: Field[Double] = Field[Double]

    override def timesl(r: Double, v: (Double, Double)): (Double, Double) = (r*v._1, r*v._2)

    override def negate(x: (Double, Double)): (Double, Double) = (-x._1, -x._2)

    override def zero: (Double, Double) = (0d, 0d)

    override def plus(x: (Double, Double), y: (Double, Double)): (Double, Double) = (x._1+y._1, x._2+y._2)
  }

}
