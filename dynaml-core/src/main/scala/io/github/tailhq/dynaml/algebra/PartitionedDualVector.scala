package io.github.tailhq.dynaml.algebra

import breeze.linalg.{DenseVector, NumericOps, Transpose}
import org.apache.spark.rdd.RDD

/**
  * A distributed row vector that is stored in blocks.
  * @param data The underlying [[RDD]] which should consist of
  *             block indices and a row vector containing
  *             all the elements in the said block.
  *
  * @author tailhq date 13/10/2016.
  *
  * */
class PartitionedDualVector(
  data: Stream[(Long, Transpose[DenseVector[Double]])],
  num_cols: Long = -1L, num_col_blocks: Long = -1L)
  extends NumericOps[PartitionedDualVector] {

  lazy val colBlocks = if(num_col_blocks == -1L) data.map(_._1).max else num_col_blocks

  lazy val rowBlocks = 1L

  lazy val cols: Long = if(num_cols == -1L) data.map(_._2.inner.length).sum.toLong else num_cols

  lazy val rows: Long = 1L

  def _data = data

  override def repr: PartitionedDualVector = this

  def t: PartitionedVector = new PartitionedVector(data.map(c => (c._1, c._2.t)), cols, colBlocks)


}

object PartitionedDualVector {

  def apply(data: Stream[(Long, Transpose[DenseVector[Double]])], num_columns: Long = -1L): PartitionedDualVector = {

    val nC = if(num_columns == -1L) data.map(_._1).max else num_columns

    new PartitionedDualVector(data.sortBy(_._1), num_cols = nC, num_col_blocks = data.length)

  }

  def apply(length: Long, numElementsPerBlock: Int, tabFunc: (Long) => Double): PartitionedDualVector = {
    val num_blocks: Long = length/numElementsPerBlock
    val blockIndices = 0L until num_blocks
    val indices = (0L until length).grouped(numElementsPerBlock).toStream

    new PartitionedDualVector(blockIndices.zip(indices).toStream.map(c =>
      (c._1, DenseVector.tabulate[Double](c._2.length)(i => tabFunc(i.toLong + c._1*numElementsPerBlock)).t)),
      num_cols = length, num_col_blocks = num_blocks
    )
  }

}
