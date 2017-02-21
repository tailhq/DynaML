package io.github.mandar2812.dynaml.analysis

import spire.algebra.{Field, InnerProductSpace}

/**
  * @author mandar2812 date: 21/02/2017.
  */
object implicits {

  implicit val innerProdDouble = new InnerProductSpace[Double, Double] {
    override def dot(v: Double, w: Double) = v*w

    override implicit def scalar = Field[Double]

    override def timesl(r: Double, v: Double) = r*v

    override def negate(x: Double) = -x

    override def zero = 0.0

    override def plus(x: Double, y: Double) = x + y
  }

}
