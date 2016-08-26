package io.github.mandar2812.dynaml.analysis

import breeze.linalg.DenseVector
import spire.algebra.{Field, NRoot}

/**
  * Created by mandar on 18/7/16.
  */
class VectorField(num_dim: Int) extends Field[DenseVector[Double]]
  with NRoot[DenseVector[Double]] {
  override def quot(a: DenseVector[Double],
                    b: DenseVector[Double]): DenseVector[Double] =
    div(a, b) - mod(a, b)

  override def gcd(a: DenseVector[Double],
                   b: DenseVector[Double]): DenseVector[Double] =
    DenseVector((a.toArray zip b.toArray).map(couple => couple._1 % couple._2))

  override def mod(a: DenseVector[Double],
                   b: DenseVector[Double]): DenseVector[Double] =
    DenseVector((a.toArray zip b.toArray).map(couple => couple._1 % couple._2))

  override def one: DenseVector[Double] = DenseVector.ones[Double](num_dim)

  override def times(x: DenseVector[Double],
                     y: DenseVector[Double]): DenseVector[Double] = x :* y

  override def negate(x: DenseVector[Double]): DenseVector[Double] = x.map(_ * -1.0)

  override def zero: DenseVector[Double] = DenseVector.zeros[Double](num_dim)

  override def div(x: DenseVector[Double],
                   y: DenseVector[Double]): DenseVector[Double] = x :/ y

  override def plus(x: DenseVector[Double],
                    y: DenseVector[Double]): DenseVector[Double] = x + y

  override def nroot(a: DenseVector[Double], n: Int): DenseVector[Double] =
    a.map(x => math.pow(x, 1.0/n.toDouble))

  override def fpow(a: DenseVector[Double], b: DenseVector[Double]): DenseVector[Double] =
    DenseVector((a.toArray zip b.toArray).map(couple => math.pow(couple._1, couple._2)))

  override def fromDouble(a: Double): DenseVector[Double] = DenseVector.fill[Double](num_dim)(a)
}

object VectorField {

  def apply(n: Int) = new VectorField(n)
}