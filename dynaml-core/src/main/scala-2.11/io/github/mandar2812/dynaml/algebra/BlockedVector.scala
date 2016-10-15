package io.github.mandar2812.dynaml.algebra

import breeze.linalg.{DenseVector, NumericOps}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
  * @author mandar2812 date 13/10/2016.
  * A distributed vector that is stored in blocks.
  *
  * @param data The underlying [[RDD]] which should consist of
  *             block indices and a breeze [[DenseVector]] containing
  *             all the elements in the said block.
  */
private[dynaml] class BlockedVector(data: RDD[(Long, DenseVector[Double])],
                                     num_rows: Long = -1L,
                                     num_row_blocks: Long = -1L)
  extends SparkVectorLike[DenseVector[Double]] with NumericOps[BlockedVector] {

  lazy val rowBlocks = if(num_row_blocks == -1L) data.keys.max else num_row_blocks

  lazy val colBlocks = 1L

  override var vector = data

  lazy val rows: Long = if(num_rows == -1L) data.map(_._2.length).sum().toLong else num_rows

  lazy val cols: Long = 1L

  def _data = vector


  override def repr: BlockedVector = this

  def t: BlockedDualVector = new BlockedDualVector(data.map(c => (c._1, c._2.t)), rows, rowBlocks)

  def persist: Unit = {
    data.persist(StorageLevel.MEMORY_AND_DISK)
  }

  def unpersist: Unit = {
    data.unpersist()
  }


}

object BlockedVector {

  /**
    * Create a [[BlockedVector]] from a [[SparkVector]], this
    * method takes the underlying key-value [[RDD]] and groups it
    * by blocks converting each block to a breeze [[DenseVector]]
    *
    * @param v The distributed vector
    * @param numElementsRowBlock Maximum number of rows in each block
    *
    */
  def apply(v: SparkVector, numElementsRowBlock: Int): BlockedVector = {
    new BlockedVector(
      v._vector.map(e => (e._1/numElementsRowBlock,e)).groupByKey().map(b => {
        val (blocIndex, locData) = (b._1, b._2.map(cp => (cp._1 - b._1*numElementsRowBlock,cp._2)).toMap)
        (blocIndex, DenseVector.tabulate[Double](locData.size)(i => locData(i)))
      })
    )
  }
}