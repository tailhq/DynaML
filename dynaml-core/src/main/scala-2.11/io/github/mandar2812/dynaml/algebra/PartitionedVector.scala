package io.github.mandar2812.dynaml.algebra

import breeze.linalg.{DenseVector, NumericOps, Transpose}
import org.apache.spark.rdd.RDD

/**
  * @author mandar2812 date 13/10/2016.
  * A distributed vector that is stored in blocks.
  * @param data The underlying [[RDD]] which should consist of
  *             block indices and a breeze [[DenseVector]] containing
  *             all the elements in the said block.
  */
private[dynaml] class PartitionedVector(data: Stream[(Long, DenseVector[Double])],
                                        num_rows: Long = -1L,
                                        num_row_blocks: Long = -1L)
  extends NumericOps[PartitionedVector] {

  lazy val rowBlocks = if(num_row_blocks == -1L) data.map(_._1).max else num_row_blocks

  lazy val colBlocks = 1L

  lazy val rows: Long = if(num_rows == -1L) data.map(_._2.length).sum.toLong else num_rows

  lazy val cols: Long = 1L

  def _data = data

  override def repr: PartitionedVector = this

  def t: PartitionedDualVector = new PartitionedDualVector(data.map(c => (c._1, c._2.t)), rows, rowBlocks)

}


object PartitionedVector {

  def apply(data: Stream[(Long, DenseVector[Double])], num_rows: Long = -1L): PartitionedVector = {

    val nC = if(num_rows == -1L) data.map(_._1).max else num_rows

    new PartitionedVector(data.sortBy(_._1), num_rows = nC, num_row_blocks = data.length)

  }

  def apply(length: Long, numElementsPerBlock: Int, tabFunc: (Long) => Double): PartitionedVector = {
    val num_blocks: Long = length/numElementsPerBlock
    val blockIndices = 0L until num_blocks
    val indices = (0L until length).grouped(numElementsPerBlock).toStream

    new PartitionedVector(blockIndices.zip(indices).toStream.map(c =>
      (c._1, DenseVector.tabulate[Double](c._2.length)(i => tabFunc(i.toLong + c._1*numElementsPerBlock)))),
      num_rows = length, num_row_blocks = num_blocks
    )
  }

}
