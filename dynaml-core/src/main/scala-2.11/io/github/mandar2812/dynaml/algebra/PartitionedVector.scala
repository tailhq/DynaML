package io.github.mandar2812.dynaml.algebra

import breeze.linalg.{DenseVector, NumericOps}
import org.apache.spark.rdd.RDD

import scala.collection.immutable.NumericRange

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

  def map(func: ((Long, DenseVector[Double])) => (Long, DenseVector[Double])): PartitionedVector =
    new PartitionedVector(data.map(func), rows, rowBlocks)

  def apply(r: NumericRange[Long]): PartitionedVector = {

    new PartitionedVector(
      data.filter(e => r.contains(e._1))
        .map(e => (e._1 - r.min, e._2)),
      num_row_blocks = r.length
    )
  }

  def toBreezeVector = DenseVector.vertcat(data.sortBy(_._1).map(_._2):_*)

  def reverse: PartitionedVector = map(c => (rowBlocks - 1L - c._1, DenseVector(c._2.toArray.reverse)))

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

  def vertcat(vectors: PartitionedVector*): PartitionedVector = {
    //sanity check
    assert(vectors.map(_.colBlocks).distinct.length == 1,
      "In case of vertical concatenation of matrices their columns sizes must be equal")

    val sizes = vectors.map(_.rowBlocks)
    new PartitionedVector(vectors.zipWithIndex.map(couple => {
      val offset = sizes.slice(0, couple._2).sum
      couple._1._data.map(c => (c._1+offset, c._2))
    }).reduce((a,b) => a.union(b)))
  }


}
