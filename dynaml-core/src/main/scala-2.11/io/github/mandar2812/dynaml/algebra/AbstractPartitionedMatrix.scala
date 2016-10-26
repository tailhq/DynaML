package io.github.mandar2812.dynaml.algebra

import breeze.linalg.NumericOps

import scala.collection.immutable.NumericRange

/**
  * @author mandar2812 date: 25/10/2016.
  */
abstract class AbstractPartitionedMatrix[T](
  data: Stream[((Long, Long), T)],
  num_row_blocks: Long = -1L,
  num_col_blocks: Long = -1L)
  extends NumericOps[AbstractPartitionedMatrix[T]] {

  lazy val rowBlocks = if(num_row_blocks == -1L) data.map(_._1._1).max + 1L else num_row_blocks

  lazy val colBlocks = if(num_col_blocks == -1L) data.map(_._1._2).max + 1L else num_col_blocks

  def _data: Stream[((Long, Long), T)] = data.sortBy(_._1)

  override def repr: AbstractPartitionedMatrix[T] = this

  def t: AbstractPartitionedMatrix[T]

  def map(f: (((Long, Long), T)) => ((Long, Long), T)): AbstractPartitionedMatrix[T]

  /**
    * Slice a blocked matrix to produce a new block matrix.
    */
  def apply(r: NumericRange[Long], c: NumericRange[Long]): AbstractPartitionedMatrix[T]

  def filterBlocks(f: ((Long, Long)) => Boolean): Stream[((Long, Long), T)] =
    _data.filter(c => f(c._1))


}
