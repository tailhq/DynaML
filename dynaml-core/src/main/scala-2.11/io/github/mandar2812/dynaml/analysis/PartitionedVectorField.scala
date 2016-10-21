package io.github.mandar2812.dynaml.analysis

import io.github.mandar2812.dynaml.algebra.PartitionedVector
import io.github.mandar2812.dynaml.algebra.PartitionedMatrixOps._
import spire.algebra.Field

/**
  * Created by mandar on 19/10/2016.
  */
class PartitionedVectorField(num_dim: Long,
                             num_elements_per_block: Int)
  extends Field[PartitionedVector] {

  override def div(x: PartitionedVector, y: PartitionedVector): PartitionedVector = x :/ y

  override def quot(a: PartitionedVector, b: PartitionedVector): PartitionedVector = div(a, b) - mod(a, b)

  override def mod(a: PartitionedVector, b: PartitionedVector): PartitionedVector = a :% b

  override def gcd(a: PartitionedVector, b: PartitionedVector): PartitionedVector = a :% b

  override def one: PartitionedVector = PartitionedVector.ones(num_dim, num_elements_per_block)

  override def negate(x: PartitionedVector): PartitionedVector = x :* -1.0

  override def zero: PartitionedVector = PartitionedVector.zeros(num_dim, num_elements_per_block)

  override def plus(x: PartitionedVector, y: PartitionedVector): PartitionedVector = x + y

  override def times(x: PartitionedVector, y: PartitionedVector): PartitionedVector = x :* y
}

object PartitionedVectorField {

  def apply(n: Long, nE: Int) = new PartitionedVectorField(n, nE)
}