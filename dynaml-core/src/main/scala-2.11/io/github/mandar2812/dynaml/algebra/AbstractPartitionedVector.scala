package io.github.mandar2812.dynaml.algebra

import breeze.linalg.NumericOps

import scala.collection.immutable.NumericRange

/**
  * @author mandar2812 date: 25/10/2016.
  */
abstract class AbstractPartitionedVector[T](
  data: Stream[(Long, T)],
  num_row_blocks: Long = -1L)
  extends NumericOps[AbstractPartitionedVector[T]] {

  lazy val rowBlocks = if(num_row_blocks == -1L) data.map(_._1).max + 1L else num_row_blocks

  lazy val colBlocks = 1L

  lazy val cols: Long = 1L

  def _data = data.sortBy(_._1)

  override def repr: AbstractPartitionedVector[T] = this

  def map(func: ((Long, T)) => (Long, T)): AbstractPartitionedVector[T]

  def apply(r: NumericRange[Long]): AbstractPartitionedVector[T]


}
