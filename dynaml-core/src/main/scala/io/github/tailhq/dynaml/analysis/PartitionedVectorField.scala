package io.github.tailhq.dynaml.analysis

import io.github.tailhq.dynaml.algebra.PartitionedVector
import io.github.tailhq.dynaml.algebra.PartitionedMatrixOps._
import spire.algebra.{Eq, Field, InnerProductSpace}
import io.github.tailhq.dynaml.analysis.implicits._

/**
  * Created by mandar on 19/10/2016.
  */
class PartitionedVectorField(
  num_dim: Long,
  num_elements_per_block: Int) extends
  Field[PartitionedVector] with
  InnerProductSpace[PartitionedVector, Double] {

  override def div(x: PartitionedVector, y: PartitionedVector): PartitionedVector = x /:/ y

  override def equot(a: PartitionedVector, b: PartitionedVector): PartitionedVector = div(a, b) - emod(a, b)

  override def emod(a: PartitionedVector, b: PartitionedVector): PartitionedVector = a %:% b

  override def gcd(a: PartitionedVector, b: PartitionedVector)(implicit ev: Eq[PartitionedVector]) = a %:% b

  override def lcm(a: PartitionedVector, b: PartitionedVector)(implicit ev: Eq[PartitionedVector]) = a %:% b

  override def one: PartitionedVector = PartitionedVector.ones(num_dim, num_elements_per_block)

  override def negate(x: PartitionedVector): PartitionedVector = x *:* -1.0

  override def zero: PartitionedVector = PartitionedVector.zeros(num_dim, num_elements_per_block)

  override def plus(x: PartitionedVector, y: PartitionedVector): PartitionedVector = x + y

  override def times(x: PartitionedVector, y: PartitionedVector): PartitionedVector = x *:* y

  override def dot(v: PartitionedVector, w: PartitionedVector) = v dot w

  override implicit def scalar = Field[Double]

  override def timesl(r: Double, v: PartitionedVector) = v*r
}

object PartitionedVectorField {

  def apply(n: Long, nE: Int) = new PartitionedVectorField(n, nE)
}

abstract class InnerProductPV extends InnerProductSpace[PartitionedVector, Double] {
  override def dot(v: PartitionedVector, w: PartitionedVector) = v dot w

  override implicit def scalar = Field[Double]

  override def timesl(r: Double, v: PartitionedVector) = v*r

  override def negate(x: PartitionedVector) = x*(-1d)

  override def plus(x: PartitionedVector, y: PartitionedVector) = x+y
}

object InnerProductPV {

  def apply(num_dim: Long, num_elements_per_block: Int) = new InnerProductPV {
    override def zero = PartitionedVector.zeros(num_dim, num_elements_per_block)
  }

  def apply(zeroElem: PartitionedVector): InnerProductPV = new InnerProductPV {
    override def zero = zeroElem
  }

}