package io.github.mandar2812.dynaml.algebra

import breeze.linalg.{DenseVector, NumericOps}
import io.github.mandar2812.dynaml.probability.RandomVariable
import org.apache.spark.rdd.RDD

import scala.collection.immutable.NumericRange

/**
  * A distributed vector that is stored in blocks.
  * @param data The underlying [[RDD]] which should consist of
  *             block indices and a breeze [[DenseVector]] containing
  *             all the elements in the said block.
  * @param num_rows Number of elements as a [[Long]], in case not specified
  *                 is calculated on instance creation.
  * @param num_row_blocks Number of blocks, in case not specified
  *                 is calculated on instance creation.
  * @author mandar2812 date 13/10/2016.
  *
  * */
class PartitionedVector(
  data: Stream[(Long, DenseVector[Double])],
  num_rows: Long = -1L,
  num_row_blocks: Long = -1L) extends
  AbstractPartitionedVector[DenseVector[Double]](data, num_row_blocks) with
  NumericOps[PartitionedVector] {

  self =>

  lazy val rows: Long = if(num_rows == -1L) data.map(_._2.length).sum.toLong else num_rows

  override lazy val cols: Long = 1L

  override def repr: PartitionedVector = this

  def t: PartitionedDualVector = new PartitionedDualVector(data.map(c => (c._1, c._2.t)), rows, rowBlocks)

  override def map(func: ((Long, DenseVector[Double])) => (Long, DenseVector[Double])): PartitionedVector =
    new PartitionedVector(data.map(func), rows, rowBlocks)

  override def apply(r: NumericRange[Long]): PartitionedVector = new PartitionedVector(
    data.filter(e => r.contains(e._1)).map(e => (e._1 - r.min, e._2)),
    num_row_blocks = r.length)
  
  def toBreezeVector = DenseVector.vertcat(data.sortBy(_._1).map(_._2):_*)

  def toStream = PartitionedVector.toStream(self)

  def reverse: PartitionedVector = map(c => (rowBlocks - 1L - c._1, DenseVector(c._2.toArray.reverse)))

}


object PartitionedVector {

  /**
    * Create a [[PartitionedVector]] given the input blocks.
    *
    * */
  def apply(data: Stream[(Long, DenseVector[Double])], num_rows: Long = -1L): PartitionedVector = {

    val nC = if(num_rows == -1L) data.map(_._2.length).sum else num_rows

    new PartitionedVector(data, num_rows = nC, num_row_blocks = data.length)

  }

  /**
    * Create a [[PartitionedVector]] from a tabulation function
    *
    * */
  def apply(length: Long, numElementsPerBlock: Int, tabFunc: (Long) => Double): PartitionedVector = {
    val num_blocks: Long = math.ceil(length.toDouble/numElementsPerBlock).toLong
    val blockIndices = 0L until num_blocks
    val indices = (0L until length).grouped(numElementsPerBlock).toStream

    new PartitionedVector(blockIndices.zip(indices).toStream.map(c =>
      (c._1, DenseVector.tabulate[Double](c._2.length)(i => tabFunc(i.toLong + c._1*numElementsPerBlock)))),
      num_rows = length, num_row_blocks = num_blocks
    )
  }

  /**
    * Create a [[PartitionedVector]] from a stream
    * @param d input stream
    * @param length The size of the stream
    * @param num_elements_per_block The size of each block
    * @return A [[PartitionedVector]] instance.
    * */
  def apply(d: Stream[Double], length: Long, num_elements_per_block: Int): PartitionedVector = {
    val num_blocks: Long = math.ceil(length.toDouble/num_elements_per_block).toLong
    val data = d.grouped(num_elements_per_block)
      .zipWithIndex
      .map(c => (c._2.toLong, DenseVector(c._1.toArray)))
      .toStream

    new PartitionedVector(data, num_rows = length, num_row_blocks = num_blocks)
  }

  /**
    * Create a [[PartitionedVector]] from a breeze [[DenseVector]]
    * @param v input vector
    * @param num_elements_per_block The size of each block
    * @return A [[PartitionedVector]] instance.
    * */
  def apply(v: DenseVector[Double], num_elements_per_block: Int): PartitionedVector = {
    val blocks = v.toArray
      .grouped(num_elements_per_block)
      .zipWithIndex
      .map(c => (c._2.toLong, DenseVector(c._1)))
      .toStream

    new PartitionedVector(blocks)
  }


  /**
    * Vertically merge a number of partitioned vectors.
    * */
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

  /**
    * Populate a partitioned vector with zeros.
    * */
  def zeros(numElements: Long, numElementsPerBlock: Int): PartitionedVector =
    PartitionedVector(numElements, numElementsPerBlock, _ => 0.0)

  /**
    * Populate a partitioned vector with ones.
    * */
  def ones(numElements: Long, numElementsPerBlock: Int): PartitionedVector =
    PartitionedVector(numElements, numElementsPerBlock, _ => 1.0)

  /**
    * Populate a partitioned vector with I.I.D samples from a
    * specified [[RandomVariable]]
    * */
  def rand(numElements: Long, numElementsPerBlock: Int, r: RandomVariable[Double]): PartitionedVector =
    PartitionedVector(numElements, numElementsPerBlock, _ => r.draw)

  /**
    * Convert a [[PartitionedVector]] to a Scala [[Stream]].
    * */
  def toStream(pvec: PartitionedVector): Stream[Double] =
    pvec._data.map(_._2.toArray.toStream).reduceLeft((a, b) => a ++ b)


}
