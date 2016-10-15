package io.github.mandar2812.dynaml.algebra

import breeze.linalg.{DenseVector, NumericOps, Transpose}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
  * @author mandar2812 date 13/10/2016.
  * A distributed row vector that is stored in blocks.
  * @param data The underlying [[RDD]] which should consist of
  *             block indices and a row vector containing
  *             all the elements in the said block.
  */
private[dynaml] class BlockedDualVector(data: RDD[(Long, Transpose[DenseVector[Double]])],
                                         num_cols: Long = -1L,
                                         num_col_blocks: Long = -1L)
  extends SparkVectorLike[Transpose[DenseVector[Double]]] with NumericOps[BlockedDualVector] {

  lazy val colBlocks = if(num_col_blocks == -1L) data.keys.max else num_col_blocks

  lazy val rowBlocks = 1L

  override var vector = data

  lazy val cols: Long = if(num_cols == -1L) data.map(_._2.inner.length).sum().toLong else num_cols

  lazy val rows: Long = 1L

  def _data = vector

  override def repr: BlockedDualVector = this

  def t: BlockedVector = new BlockedVector(data.map(c => (c._1, c._2.t)), cols, colBlocks)

  def persist: Unit = {
    data.persist(StorageLevel.MEMORY_AND_DISK)
  }

  def unpersist: Unit = {
    data.unpersist()
  }


}
