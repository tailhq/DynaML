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

import breeze.generic.UFunc
import breeze.linalg.{DenseMatrix, cholesky, inv}
import org.apache.spark.rdd.RDD
import BlockedMatrixOps._
import PartitionedMatrixOps._

/**
  * @author mandar2812 date: 12/10/2016.
  *
  * Implementation of blocked version of
  * the Cholesky decomposition algorithm
  * Reference [[http://www.cs.utexas.edu/users/flame/Notes/NotesOnCholReal.pdf]]
  */
object bcholesky extends UFunc {

  def choleskyAcc(mat: SparkBlockedMatrix,
                  nAcc: Long,
                  mAcc: RDD[((Long, Long), DenseMatrix[Double])])
  : RDD[((Long, Long), DenseMatrix[Double])] =
    mat.colBlocks*mat.rowBlocks match {

      /*
      * Incase the matrix consists only of one partitioned block
      * calculate its Cholesky decomposition and return it as the
      * result.
      * */
      case 1L => mAcc.union(mat._data.map(f => ((f._1._1 + nAcc, f._1._2 + nAcc), cholesky(f._2))))

      /*
      * Incase the matrix contains more than one block:
      *
      * A = A_11 | *
      *     ----------
      *     A_21 | A_22
      *
      * 1. Take the first block, calculate its Cholesky decomposition and store it in L_11
      * 2. Set L_21 = A_21*L_11^-T
      * 3. Set L_22 = Cholesky(A_22 - L_21*L_21^T)
      * */
      case _ =>
        val L_11: SparkBlockedMatrix =
          mat(0L to 0L, 0L to 0L).map(t => (t._1, cholesky(t._2)))

        val tFunc: (((Long, Long), DenseMatrix[Double])) => ((Long, Long), DenseMatrix[Double]) =
          c => ((c._1._1+nAcc, c._1._2+nAcc), c._2)

        val L_21: SparkBlockedMatrix = mat(1L until mat.rowBlocks, 0L to 0L)*L_11.map(t => (t._1, inv(t._2))).t

        L_21.persist
        L_11.persist

        choleskyAcc(
          mat(1L until mat.rowBlocks, 1L until mat.colBlocks) - (L_21*L_21.t),
          nAcc+1L,
          mAcc.union(L_11.map(t => ((t._1._1+nAcc, t._1._2+nAcc), t._2))._data)
            .union(L_21.map(tFunc)._data)
        )

    }

  def choleskyPAcc(mat: PartitionedMatrix,
                   nAcc: Long,
                   mAcc: Stream[((Long, Long), DenseMatrix[Double])])
  : Stream[((Long, Long), DenseMatrix[Double])] =
    mat.colBlocks*mat.rowBlocks match {

      /*
      * Incase the matrix consists only of one partitioned block
      * calculate its Cholesky decomposition and return it as the
      * result.
      * */
      case 1L => mAcc ++ mat._data.map(f => ((f._1._1 + nAcc, f._1._2 + nAcc), cholesky(f._2)))

      /*
      * Incase the matrix contains more than one block:
      *
      * A = A_11 | *
      *     ----------
      *     A_21 | A_22
      *
      * 1. Take the first block, calculate its Cholesky decomposition and store it in L_11
      * 2. Set L_21 = A_21*L_11^-T
      * 3. Set L_22 = Cholesky(A_22 - L_21*L_21^T)
      * */
      case _ =>
        val L_11: PartitionedMatrix =
          mat(0L to 0L, 0L to 0L).map(t => (t._1, cholesky(t._2)))

        val tFunc: (((Long, Long), DenseMatrix[Double])) => ((Long, Long), DenseMatrix[Double]) =
          c => ((c._1._1+nAcc+1, c._1._2+nAcc), c._2)

        val L_21: PartitionedMatrix =  mat(1L until mat.rowBlocks, 0L to 0L) * L_11.map(t => (t._1, inv(t._2))).t


        choleskyPAcc(
          mat(1L until mat.rowBlocks, 1L until mat.colBlocks) - (L_21*L_21.t),
          nAcc+1L,
          mAcc ++ L_11._data.map(t => ((t._1._1+nAcc, t._1._2+nAcc), t._2)) ++ L_21._data.map(tFunc)
        )

    }

  implicit object ImplPCholesky_DM extends Impl[PartitionedPSDMatrix, LowerTriPartitionedMatrix] {
    def apply(X: PartitionedPSDMatrix): LowerTriPartitionedMatrix = {
      //Sanity Checks
      require(X.rows == X.cols, "For performing a Cholesky decomposition, the blocked matrix must be square")
      require(X.rowBlocks == X.colBlocks,
        "For performing a Cholesky decomposition the blocked matrix must be partitioned equally in rows and columns")

      /*
      * Call tail recursive routine.
      *
      * For the upper triangular blocks of the matrix,
      * replace them with zero matrices thus mAcc starts
      * with these zero matrix blocks.
      * */
      val dat = choleskyPAcc(
        X, 0L, Stream()
        /*X._data
          .filter(c => c._1._2 > c._1._1)
          .map(c => (c._1, DenseMatrix.zeros[Double](c._2.rows, c._2.cols)))*/
      ).sortBy(_._1)

      /*
      * Create new blocked matrix from accumulated Stream
      * */
      new LowerTriPartitionedMatrix(dat, X.rows, X.cols, X.rowBlocks, X.colBlocks)
    }
  }

  implicit object ImplCholesky_DM extends Impl[SparkBlockedMatrix, SparkBlockedMatrix] {
    def apply(X: SparkBlockedMatrix): SparkBlockedMatrix = {
      //Sanity Checks
      assert(X.rows == X.cols, "For performing a Cholesky decomposition, the blocked matrix must be square")
      assert(X.rowBlocks == X.colBlocks,
        "For performing a Cholesky decomposition the blocked matrix must be partitioned equally in rows and columns")

      /*
      * Call tail recursive routine.
      *
      * For the upper triangular blocks of the matrix,
      * replace them with zero matrices thus mAcc starts
      * with these zero matrix blocks.
      * */
      val dat = choleskyAcc(
        X, 0L,
        X._data
          .filter(c => c._1._2 > c._1._1)
          .map(c => (c._1, DenseMatrix.zeros[Double](c._2.rows, c._2.cols)))
      )

      /*
      * Create new blocked matrix from accumulated RDD
      * */
      new SparkBlockedMatrix(dat, X.rows, X.cols, X.rowBlocks, X.colBlocks)
    }
  }
}
