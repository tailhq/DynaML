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
import org.apache.spark.rdd.RDD

import scala.collection.immutable.NumericRange

/**
  *  @author mandar2812 date: 28/09/2016
  *
  *  A distributed matrix backed by a spark [[RDD]]
  *
  */
class SparkMatrix(baseMatrix: RDD[((Long, Long), Double)]) extends NumericOps[SparkMatrix] {

  lazy val rows = baseMatrix.map(_._1._1).max() + 1L

  lazy val cols = baseMatrix.map(_._1._2).max() + 1L

  //Perform sanity checks
  assert(baseMatrix.keys.distinct.count == rows*cols, "Matrix Indices must be unique")
  assert(baseMatrix.map(_._1._1).min() == 0L && baseMatrix.map(_._1._2).min() == 0L && rows > 0L && cols > 0L,
  "Row and column indices must be between 0 -> N-1")

  protected val matrix: RDD[((Long, Long), Double)] = baseMatrix

  override def repr: SparkMatrix = this

  /**
    * @return The backing [[RDD]]
    */
  def _matrix = matrix

  /**
    * Obtain transpose of the matrix.
    *
    */
  def t: SparkMatrix = new SparkMatrix(this.baseMatrix.map(c => ((c._1._2, c._1._1), c._2)))

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

}

/**
  * @author mandar2812 date: 09/10/2016
  *
  * A distributed square matrix backed by a spark [[RDD]]
  *
  */
class SparkSquareMatrix(baseMatrix: RDD[((Long, Long), Double)]) extends SparkMatrix(baseMatrix) {

  //Sanity Checks
  assert(rows == cols, "For a square matrix, rows must be equal to columns")

  /**
    * Extract diagonal elements into a new [[SparkSquareMatrix]]
    *
    */
  def diag: SparkSquareMatrix = new SparkSquareMatrix(
    baseMatrix.map(c => if(c._1._1 == c._1._2) c else (c._1, 0.0))
  )

  /**
    * Extract lower triangular elements into a new [[SparkSquareMatrix]]
    */
  def L: SparkSquareMatrix = new SparkSquareMatrix(
    baseMatrix.map(c => if(c._1._1 <= c._1._2) c else (c._1, 0.0))
  )

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
class SparkPSDMatrix(basePSDMat: RDD[((Long, Long), Double)])
  extends SparkSquareMatrix(baseMatrix = basePSDMat.flatMap(e => {
    //If element is diagonal just emit otherwise reflect indices and emit
    if(e._1._1 == e._1._1) Seq(e) else Seq(e, (e._1.swap, e._2))
  })) {
  //Carry out sanity checks to prevent obvious errors from non PSD arguments

  assert(
    this.diag._matrix.filter(e => e._1._1 == e._1._2 && e._2 > 0.0).count() == rows,
    "All diagonal elements must be positive !!")
}