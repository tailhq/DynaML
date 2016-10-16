/*
Copyright 2016 Mandar Chandorkar

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
* */
package io.github.mandar2812.dynaml.algebra

import breeze.linalg.{DenseMatrix, NumericOps}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.immutable.NumericRange

/**
  * @author mandar2812 date 13/10/2016.
  * A distributed matrix that is stored in blocks.
  *
  * @param data The underlying [[RDD]] which should consist of
  *             block indices and a breeze [[DenseMatrix]] containing
  *             all the elements in the said block.
  */
private[dynaml] class BlockedMatrix(data: RDD[((Long, Long), DenseMatrix[Double])],
                                     num_rows: Long = -1L, num_cols: Long = -1L,
                                     num_row_blocks: Long = -1L, num_col_blocks: Long = -1L)
  extends NumericOps[BlockedMatrix] {

  lazy val rowBlocks = if(num_row_blocks == -1L) data.keys.map(_._1).max else num_row_blocks

  lazy val colBlocks = if(num_col_blocks == -1L) data.keys.map(_._2).max else num_col_blocks


  lazy val rows: Long = if(num_rows == -1L) data.filter(_._1._2 == 0L).map(_._2.rows).sum().toLong else num_rows

  lazy val cols: Long = if(num_cols == -1L) data.filter(_._1._1 == 0L).map(_._2.cols).sum().toLong else num_cols

  def _data = data


  override def repr: BlockedMatrix = this

  /**
    * Transpose of blocked matrix
    * */
  def t: BlockedMatrix = new BlockedMatrix(
    data.map(c => (c._1.swap, c._2.t)),
    cols, rows, colBlocks, rowBlocks)

  /**
    * Persist blocked matrix in memory
    */
  def persist: Unit = {
    data.persist(StorageLevel.MEMORY_AND_DISK)
  }

  def unpersist: Unit = {
    data.unpersist()
  }

  def map(f: (((Long, Long), DenseMatrix[Double])) => ((Long, Long), DenseMatrix[Double])): BlockedMatrix =
    new BlockedMatrix(data.map(f), rows, cols, rowBlocks, colBlocks)

  /**
    * Slice a blocked matrix to produce a new block matrix.
    */
  def apply(r: NumericRange[Long], c: NumericRange[Long]): BlockedMatrix = {

    new BlockedMatrix(
      data.filter(e => r.contains(e._1._1) && c.contains(e._1._2))
        .map(e => ((e._1._1 - r.min, e._1._2 - c.min), e._2)),
      num_row_blocks = r.length, num_col_blocks = c.length
    )
  }

  def apply(f: ((Long, Long)) => Boolean): RDD[((Long, Long), DenseMatrix[Double])] =
      data.filter(c => f(c._1))

}

object BlockedMatrix {

  /**
    * Create a [[BlockedMatrix]] from a [[SparkMatrix]], this
    * method takes the underlying key-value [[RDD]] and groups it
    * by blocks converting each block to a breeze [[DenseMatrix]]
    *
    * @tparam T The type of the [[SparkMatrix]] instance
    *
    * @param m The distributed matrix
    * @param numElementsRowBlock Maximum number of rows in each block
    * @param numElementsColBlock Maximum number of columns in each block matrix
    *
    */
  def apply[T <: SparkMatrix](m: T, numElementsRowBlock: Int, numElementsColBlock: Int): BlockedMatrix = {
    new BlockedMatrix(
      m._matrix
        .map(e => ((e._1._1/numElementsRowBlock, e._1._2/numElementsColBlock),e))
        .groupByKey().map(c => {

        //Construct local block matrices
        val (blocIndex, locData) = (
          c._1,
          c._2.map(el =>
            ((
              el._1._1 - c._1._1*numElementsRowBlock,
              el._1._2 - c._1._2*numElementsColBlock),el._2)).toMap
          )

        (blocIndex, DenseMatrix.tabulate[Double](
          locData.keys.map(_._1).max.toInt + 1,
          locData.keys.map(_._2).max.toInt + 1)((i,j) => locData(i,j)))
      }),
      num_rows = m.rows, num_cols = m.cols,
      num_row_blocks = m.rows/numElementsRowBlock,
      num_col_blocks = m.cols/numElementsColBlock
    )
  }

  def vertcat(vectors: BlockedMatrix*): BlockedMatrix = {
    //sanity check
    assert(vectors.map(_.colBlocks).distinct.length == 1,
      "In case of vertical concatenation of matrices their columns sizes must be equal")

    val sizes = vectors.map(_.rowBlocks)
    new BlockedMatrix(vectors.zipWithIndex.map(couple => {
      val offset = sizes.slice(0, couple._2).sum
      couple._1._data.map(c => ((c._1._1+offset, c._1._2), c._2))
    }).reduce((a,b) => a.union(b)))
  }

  def horzcat(vectors: BlockedMatrix*): BlockedMatrix = {
    //sanity check
    assert(vectors.map(_.rowBlocks).distinct.length == 1,
      "In case of horizontal concatenation of matrices their row sizes must be equal")

    val sizes = vectors.map(_.colBlocks)
    new BlockedMatrix(vectors.zipWithIndex.map(couple => {
      val offset = sizes.slice(0, couple._2).sum
      couple._1._data.map(c => ((c._1._1, c._1._2+offset), c._2))
    }).reduce((a,b) => a.union(b)))
  }


}
