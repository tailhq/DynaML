/*
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

import breeze.linalg.NumericOps
import io.github.mandar2812.dynaml.kernels.Kernel
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.immutable.NumericRange

/**
  *  @author mandar2812 date: 28/09/2016
  *
  *  A distributed matrix backed by a spark [[RDD]]
  *
  */
class SparkMatrix(baseMatrix: RDD[((Long, Long), Double)],
                  num_rows: Long = -1L, num_cols: Long = -1L,
                  sanityChecks: Boolean = true) extends NumericOps[SparkMatrix] {

  lazy val rows = if(num_rows == -1L) baseMatrix.map(_._1._1).max() + 1L else num_rows

  lazy val cols = if(num_cols == -1L) baseMatrix.map(_._1._2).max() + 1L else num_cols

  if(sanityChecks) {
    //Perform sanity checks
    require(baseMatrix.keys.distinct.count == rows*cols,
      "Matrix Indices must be unique")
    require(
      baseMatrix.map(_._1._1).min() == 0L && baseMatrix.map(_._1._2).min() == 0L && rows > 0L && cols > 0L,
      "Row and column indices must be between 0 -> N-1")
  }

  protected var matrix: RDD[((Long, Long), Double)] = baseMatrix

  override def repr: SparkMatrix = this

  /**
    * @return The backing [[RDD]]
    */
  def _matrix = matrix

  /**
    * Obtain transpose of the matrix.
    *
    */
  def t: SparkMatrix = new SparkMatrix(_matrix.map(c => ((c._1._2, c._1._1), c._2)))

  /**
    * Get a sub-matrix based on a range of rows and columns
    *
    */
  def apply(r: NumericRange[Long], c: NumericRange[Long]): SparkMatrix =
    new SparkMatrix(
      matrix.filterByRange((r.min, c.min), (r.max, c.max))
        .map(e => ((e._1._1 - r.min, e._1._2 - c.min), e._2))
    )

  def apply(r: Range, c: Range): SparkMatrix =
    new SparkMatrix(
      matrix.filterByRange((r.min, c.min), (r.max, c.max))
        .map(e => ((e._1._1 - r.min, e._1._2 - c.min), e._2))
    )

  def persist: Unit = {
    matrix.persist(StorageLevel.MEMORY_AND_DISK)
  }

  def unpersist: Unit = {
    matrix.unpersist()
  }

}

object SparkMatrix {

  /**
    * Populate a matrix using a DynaML [[Kernel]] instance
    * applied on pairs of instances contained in the data.
    *
    * @tparam T The type of input features of the data
    *
    */
  def apply[T](data1: RDD[(Long, T)], data2: RDD[(Long, T)])(kernel: Kernel[T, Double]) =
    new SparkMatrix(data1.cartesian(data2).map(c => ((c._1._1, c._2._1), kernel.evaluate(c._1._2, c._2._2))))


  /**
    * Populate a matrix defined as M(i,j) = ev(i,j)
    */
  def apply(data1: RDD[Long], data2: RDD[Long])(ev: (Long, Long) => Double) =
    new SparkMatrix(data1.cartesian(data2).map(c => (c, ev(c._1, c._2))))


  def vertcat(vectors: SparkMatrix*): SparkMatrix = {
    //sanity check
    require(vectors.map(_.cols).distinct.length == 1,
      "In case of vertical concatenation of matrices their columns sizes must be equal")

    val sizes = vectors.map(_.rows)
    new SparkMatrix(vectors.zipWithIndex.map(couple => {
      val offset = sizes.slice(0, couple._2).sum
      couple._1._matrix.map(c => ((c._1._1+offset, c._1._2), c._2))
    }).reduce((a,b) => a.union(b)))
  }

  def horzcat(vectors: SparkMatrix*): SparkMatrix = {
    //sanity check
    require(vectors.map(_.rows).distinct.length == 1,
      "In case of horizontal concatenation of matrices their row sizes must be equal")

    val sizes = vectors.map(_.cols)
    new SparkMatrix(vectors.zipWithIndex.map(couple => {
      val offset = sizes.slice(0, couple._2).sum
      couple._1._matrix.map(c => ((c._1._1, c._1._2+offset), c._2))
    }).reduce((a,b) => a.union(b)))
  }


}

/**
  * @author mandar2812 date: 09/10/2016
  *
  * A distributed square matrix backed by a spark [[RDD]]
  *
  */
class SparkSquareMatrix(baseSqMatrix: RDD[((Long, Long), Double)],
                        num_rows: Long = -1L,
                        sanityChecks: Boolean = true)
  extends SparkMatrix(
    baseMatrix = baseSqMatrix,
    num_rows, num_rows, sanityChecks) {

  if(sanityChecks) {
    //Sanity Checks
    require(rows == cols, "For a square matrix, rows must be equal to columns")
  }

  /**
    * Extract diagonal elements into a new [[SparkSquareMatrix]]
    *
    */
  def diag: SparkSquareMatrix = new SparkSquareMatrix(
    _matrix.map(c => if(c._1._1 == c._1._2) c else (c._1, 0.0))
  )

  /**
    * Extract lower triangular elements into a new [[SparkSquareMatrix]]
    */
  def L: SparkSquareMatrix = new SparkSquareMatrix(
    _matrix.map(c => if(c._1._1 <= c._1._2) c else (c._1, 0.0))
  )

}

object SparkSquareMatrix {

  /**
    * Populate a square matrix defined as M(i,j) = ev(i,j)
    */
  def apply(data: RDD[Long])(ev: (Long, Long) => Double) =
  new SparkSquareMatrix(data.cartesian(data)
    .map(c => (c, ev(c._1, c._2))))

}


/**
  * @author mandar2812 date: 12/10/2016
  *
  * A distributed square positive semi-definite matrix backed by a
  * spark [[org.apache.spark.rdd.RDD]] by convention this stores only
  * the lower triangular portion exploiting the symmetry in the matrix
  * structure.
  *
  * A set of sanity checks are run during object creation
  * to confirm (loosely) the positive semi-definiteness of the matrix.
  *
  */
class SparkPSDMatrix(basePSDMat: RDD[((Long, Long), Double)],
                     num_rows: Long = -1L,
                     sanityChecks: Boolean = true)
  extends SparkSquareMatrix(baseSqMatrix = basePSDMat.flatMap(e => {
    //If element is diagonal just emit otherwise reflect indices and emit
    if(e._1._1 == e._1._2) Seq(e) else Seq(e, (e._1.swap, e._2))
  }), num_rows, sanityChecks) {
  //Carry out sanity checks to prevent obvious errors from non PSD arguments

  if(sanityChecks) {
    require(
      this.diag._matrix.filter(e => e._1._1 == e._1._2 && e._2 > 0.0).count() == rows,
      "All diagonal elements must be positive !!")
  }
}

object SparkPSDMatrix {

  /**
    * Populate a positive semi-definite matrix using a DynaML [[Kernel]] instance
    * applied on pairs of instances contained in the data.
    *
    * @tparam T The type of input features of the data
    *
    */
  def apply[T](data: RDD[(Long, T)])(kernel: (T, T) => Double) = {
    new SparkPSDMatrix(
      data.cartesian(data)
        .map(c => ((c._1._1, c._2._1), kernel(c._1._2, c._2._2)))
        .filter(e => e._1._1 >= e._1._2),
      sanityChecks = false
    )
  }

  /**
    * Populate a positive semi-definite matrix defined as M(i,j) = ev(i,j)
    *
    */
  /*def apply(data: RDD[Long])(ev: (Long, Long) => Double) =
    new SparkPSDMatrix(data.cartesian(data)
      .map(c => (c, ev(c._1, c._2)))
      .filter(e => e._1._1 >= e._1._2),
      sanityChecks = false)*/


}