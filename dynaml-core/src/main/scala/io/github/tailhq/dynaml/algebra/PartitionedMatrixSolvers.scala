package io.github.tailhq.dynaml.algebra

import breeze.generic.UFunc
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.linalg.operators.OpSolveMatrixBy
import io.github.tailhq.dynaml.utils
import io.github.tailhq.dynaml.algebra.PartitionedMatrixOps._
import io.github.tailhq.dynaml.algebra.BlockedMatrixOps._

/**
  * @author tailhq date: 21/10/2016.
  * Implementations of the `\` operator for [[PartitionedMatrix]] objects.
  */
object PartitionedMatrixSolvers extends UFunc {

  /*
  * Linear solve operators for partitioned matrices
  * */

  /**
    * Tail recursive linear solve: x = L \ y
    * @param X A lower triangular matrix (outputted by the [[bcholesky]] function)
    * @param y A partitioned vector
    * @param lAcc An accumulator storing the result block by block
    * @param acc A vector containing values to add into the result
    *
    * */
  def recLTriagSolve(X: LowerTriPartitionedMatrix,
                     y: PartitionedVector,
                     lAcc: Stream[PartitionedVector],
                     acc: PartitionedVector): PartitionedVector =
  X.colBlocks*X.rowBlocks match {
    case 1L =>
      val vSolved: DenseVector[Double] =
        X(0L to 0L, 0L to 0L)._data.head._2 \ (y(0L to 0L)-acc(0L to 0L))._data.head._2
      val vectorBlocks = lAcc++ Stream(new PartitionedVector(Stream((0L, vSolved))))

      PartitionedVector.vertcat(vectorBlocks:_*)

    case _ =>
      val (l_hh, l_rh, l_rr) = (
        X(0L to 0L, 0L to 0L),
        X(1L until X.rowBlocks, 0L to 0L),
        new LowerTriPartitionedMatrix(
          X.filterBlocks(c =>
            c._1 >= 1L && c._1 < X.rowBlocks &&
              c._2 >= 1L && c._2 < X.colBlocks && c._1 <= c._2)
            .map(c => ((c._1._1 - 1L, c._1._2-1L), c._2)),
          num_row_blocks = X.rowBlocks - 1L,
          num_col_blocks = X.colBlocks - 1L)
        )

      val (y_h, y_r) = (y(0L to 0L), y(1L until y.rowBlocks))
      val (acc_h, acc_r) = (acc(0L to 0L), acc(1L until acc.rowBlocks))

      val vSolved: DenseVector[Double] = l_hh._data.head._2 \ (y_h - acc_h)._data.head._2

      recLTriagSolve(
        l_rr, y_r,
        lAcc ++ Stream(new PartitionedVector(Stream((0L, vSolved)))),
        acc_r + PartitionedVector(l_rh._data.map(co => (co._1._1, co._2 * vSolved))))
  }


  /**
    * Tail recursive linear solve: x = U \ y
    * @param X An upper triangular matrix (transpose of matrix outputted by the [[bcholesky]] function)
    * @param y A partitioned vector
    * @param lAcc An accumulator storing the result block by block
    * @param acc A auxilarry vector containing historical values to add into the result
    *
    * */
  def recUTriagSolve(X: UpperTriPartitionedMatrix,
                     y: PartitionedVector,
                     lAcc: Stream[PartitionedVector],
                     acc: PartitionedVector): PartitionedVector =
  X.colBlocks*X.rowBlocks match {
    case 1L =>
      val vSolved: DenseVector[Double] =
        X(0L to 0L, 0L to 0L)._data.head._2 \ (y(0L to 0L) - acc(0L to 0L))._data.head._2
      val vectorBlocks = lAcc ++ Stream(new PartitionedVector(Stream((0L, vSolved))))

      PartitionedVector.vertcat(vectorBlocks.reverse:_*)

    case _ =>
      val (u_hh, u_rh, u_rr) = (
        X(X.rowBlocks-1L to X.rowBlocks-1L, X.colBlocks-1L to X.colBlocks-1L),
        X(0L until X.rowBlocks-1L, X.colBlocks-1 to X.colBlocks-1),
        new UpperTriPartitionedMatrix(
          X._underlyingdata
            .filter(c =>
              c._1._1 >= 0L && c._1._1 < X.rowBlocks-1L &&
                c._1._2 >= 0L && c._1._2 < X.colBlocks-1L),
          num_row_blocks = X.rowBlocks - 1L,
          num_col_blocks = X.colBlocks - 1L)
        )

      val (y_h, y_r) = (y(y.rowBlocks-1L to y.rowBlocks-1L), y(0L until y.rowBlocks-1L))
      val (acc_h, acc_r) = (acc(acc.rowBlocks-1L to acc.rowBlocks-1L), acc(0L until acc.rowBlocks-1))

      val vSolved: DenseVector[Double] = u_hh._data.head._2 \ (y_h - acc_h)._data.head._2

      val blocks = u_rh._data.map(co => (co._1._1, co._2*vSolved))

      recUTriagSolve(
        u_rr, y_r,
        lAcc ++ Stream(new PartitionedVector(Stream((0L, vSolved)))),
        acc_r+ PartitionedVector(blocks))
  }


  /**
    * Tail recursive linear solve: X = L \ Y
    * @param X A lower triangular matrix (outputted by the [[bcholesky]] function)
    * @param y A partitioned matrix
    * @param lAcc An accumulator storing the result block by block
    * @param acc A matrix containing values to add into the result
    *
    * */
  def recLTriagMultiSolve(X: LowerTriPartitionedMatrix,
                          y: PartitionedMatrix,
                          lAcc: Stream[PartitionedMatrix],
                          acc: PartitionedMatrix): PartitionedMatrix =
  X.colBlocks*X.rowBlocks match {
    case 1L =>
      val X_hh = X(0L to 0L, 0L to 0L)._data.head._2
      val y_hh = y(0L to 0L, 0L until y.colBlocks)
      val acc_hh = acc(0L to 0L, 0L until acc.colBlocks)
      val vSolved = (y_hh-acc_hh).map(c => (c._1, X_hh\c._2))

      val vectorBlocks = lAcc ++ Stream(vSolved)

      PartitionedMatrix.vertcat(vectorBlocks:_*)

    case _ =>
      val (l_hh, l_rh, l_rr) = (
        X(0L to 0L, 0L to 0L)._data.head._2,
        X(1L until X.rowBlocks, 0L to 0L),
        new LowerTriPartitionedMatrix(
          X._underlyingdata
            .filter(c =>
              c._1._1 >= 1L && c._1._1 < X.rowBlocks &&
                c._1._2 >= 1L && c._1._2 < X.colBlocks)
            .map(c => ((c._1._1 - 1L, c._1._2-1L), c._2)),
          num_row_blocks = X.rowBlocks - 1L,
          num_col_blocks = X.colBlocks - 1L)
        )

      val (y_h, y_r) = (
        y(0L to 0L,0L until y.colBlocks),
        y(1L until y.rowBlocks, 0L until y.colBlocks))

      val (acc_h, acc_r) = (
        acc(0L to 0L, 0L until acc.colBlocks),
        acc(1L until acc.rowBlocks, 0L until acc.colBlocks))

      val vSolved = (y_h-acc_h).map(c => (c._1, l_hh\c._2))

      val accMult = new PartitionedMatrix(utils.combine(Seq(l_rh._data, vSolved._data)).map(tuple => {
        val first = tuple.head
        val second = tuple.last
        ((first._1._1, second._1._2), first._2*second._2)
      }).toStream)

      recLTriagMultiSolve(
        l_rr, y_r,
        lAcc ++ Stream(vSolved),
        acc_r + accMult)
  }


  /**
    * Tail recursive linear solve: X = U \ Y
    * @param X A lower triangular matrix (outputted by the [[bcholesky]] function)
    * @param y A partitioned matrix
    * @param lAcc An accumulator storing the result block by block
    * @param acc A auxillary matrix containing historical values to add into the result
    *
    * */
  def recUTriagMultiSolve(X: UpperTriPartitionedMatrix,
                          y: PartitionedMatrix,
                          lAcc: Stream[PartitionedMatrix],
                          acc: PartitionedMatrix): PartitionedMatrix =
  X.colBlocks*X.rowBlocks match {
    case 1L =>

      val X_hh = X(0L to 0L, 0L to 0L)._data.head._2
      val y_hh = y(0L to 0L, 0L until y.colBlocks)
      val acc_hh = acc(0L to 0L, 0L until acc.colBlocks)
      val vSolved = (y_hh-acc_hh).map(c => (c._1, X_hh\c._2))

      val vectorBlocks = lAcc ++ Stream(vSolved)

      PartitionedMatrix.vertcat(vectorBlocks.reverse:_*)

    case _ =>
      val (u_hh, u_rh, u_rr) = (
        X(X.rowBlocks-1L to X.rowBlocks-1L, X.colBlocks-1L to X.colBlocks-1L)._data.head._2,
        X(0L until X.rowBlocks-1L, X.colBlocks-1 to X.colBlocks-1),
        new UpperTriPartitionedMatrix(
          X._underlyingdata
            .filter(c =>
              c._1._1 >= 0L && c._1._1 < X.rowBlocks-1L &&
                c._1._2 >= 0L && c._1._2 < X.colBlocks-1L),
          num_row_blocks = X.rowBlocks - 1L,
          num_col_blocks = X.colBlocks - 1L)
        )

      val (y_h, y_r) = (
        y(y.rowBlocks-1L to y.rowBlocks-1L, 0L until y.colBlocks),
        y(0L until y.rowBlocks-1L, 0L until y.colBlocks))

      val (acc_h, acc_r) = (
        acc(acc.rowBlocks-1L to acc.rowBlocks-1L, 0L until acc.colBlocks),
        acc(0L until acc.rowBlocks-1, 0L until acc.colBlocks))

      val vSolved = (y_h-acc_h).map(c => (c._1, u_hh\c._2))

      val accMult = new PartitionedMatrix(utils.combine(Seq(u_rh._data, vSolved._data)).map(tuple => {
        val first = tuple.head
        val second = tuple.last
        ((first._1._1, second._1._2), first._2*second._2)
      }).toStream)


      recUTriagMultiSolve(
        u_rr, y_r,
        lAcc ++ Stream(vSolved + acc_h),
        acc_r+ accMult)
  }


  /**
    * Tail recursive linear solve: x = L \ y
    * param X A lower triangular matrix (outputted by the [[bcholesky]] function)
    * param y A partitioned vector
    * param lAcc An accumulator storing the result block by block
    * param acc A vector containing values to add into the result
    *
    * */
/*
  def recLTriagRDDSolve(
    X: LowerTriSparkMatrix,
    y: SparkBlockedVector,
    lAcc: Stream[SparkBlockedVector],
    acc: SparkBlockedVector): PartitionedVector =
  X.colBlocks*X.rowBlocks match {
    case 1L =>
      val yMod: SparkBlockedVector = y(0L to 0L) - acc(0L to 0L)
      val vSolved: DenseVector[Double] =
        X(0L to 0L, 0L to 0L)._data.collect().head._2 \ yMod._data.collect().head._2
      val vectorBlocks = lAcc++ Stream(new PartitionedVector(Stream((0L, vSolved))))

      SparkBlockedVector.vertcat(vectorBlocks:_*)

    case _ =>
      val (l_hh, l_rh, l_rr) = (
        X(0L to 0L, 0L to 0L),
        X(1L until X.rowBlocks, 0L to 0L),
        new LowerTriSparkMatrix(
          X.filterBlocks(c =>
            c._1 >= 1L && c._1 < X.rowBlocks &&
              c._2 >= 1L && c._2 < X.colBlocks && c._1 <= c._2)
            .map(c => ((c._1._1 - 1L, c._1._2-1L), c._2)),
          num_row_blocks = X.rowBlocks - 1L,
          num_col_blocks = X.colBlocks - 1L)
        )

      val (y_h, y_r) = (y(0L to 0L), y(1L until y.rowBlocks))
      val (acc_h, acc_r) = (acc(0L to 0L), acc(1L until acc.rowBlocks))
      val yMod: SparkBlockedVector = y_h - acc_h
      val vSolved: DenseVector[Double] = l_hh._data.collect().head._2 \ yMod._data.collect().head._2

      recLTriagRDDSolve(
        l_rr, y_r,
        lAcc ++ Stream(new SparkBlockedVector(l_hh._data..parallelize((0L, vSolved)))),
        acc_r + SparkBlockedVector(l_rh._data.map(co => (co._1._1, co._2 * vSolved))))
  }
*/


  implicit object implOpSolveLowerTriPartitionedMatrixByVector
    extends OpSolveMatrixBy.Impl2[LowerTriPartitionedMatrix, PartitionedVector, PartitionedVector] {

    override def apply(A: LowerTriPartitionedMatrix, V: PartitionedVector): PartitionedVector = {
      require(A.rows == V.rows && A.cols == V.rows, "Non-conformant matrix-vector sizes")
      require(A.colBlocks == V.rowBlocks && A.rowBlocks == A.rowBlocks, "Non-conformant matrix-vector partitions")

      recLTriagSolve(A, V, Stream(), V.map(c => (c._1, DenseVector.zeros[Double](c._2.length))))
    }
  }

  implicit object implOpSolvePartitionedMatrixByVector
    extends OpSolveMatrixBy.Impl2[PartitionedMatrix, PartitionedVector, PartitionedVector] {

    override def apply(A: PartitionedMatrix, V: PartitionedVector): PartitionedVector = {
      require(A.rows == A.cols, "Matrix must be square")
      require(A.rows == V.rows && A.cols == V.rows, "Non-conformant matrix-vector sizes")
      require(A.colBlocks == V.rowBlocks && A.rowBlocks == A.rowBlocks, "Non-conformant matrix-vector partitions")

      val(lmat, umat): (LowerTriPartitionedMatrix, UpperTriPartitionedMatrix) = bLU(A)
      val z = recLTriagSolve(lmat, V, Stream(), V.map(c => (c._1, DenseVector.zeros[Double](c._2.length))))
      recUTriagSolve(umat, z, Stream(), z.map(c => (c._1, DenseVector.zeros[Double](c._2.length))))
    }
  }

  implicit object implOpSolvePartitionedMatrixByMatrix
    extends OpSolveMatrixBy.Impl2[PartitionedMatrix, PartitionedMatrix, PartitionedMatrix] {

    override def apply(A: PartitionedMatrix, V: PartitionedMatrix): PartitionedMatrix = {
      require(A.rows == A.cols, "Matrix must be square")
      require(A.rows == V.rows && A.cols == V.rows, "Non-conformant matrix-vector sizes")
      require(A.colBlocks == V.rowBlocks && A.rowBlocks == A.rowBlocks, "Non-conformant matrix-vector partitions")

      val(lmat, umat): (LowerTriPartitionedMatrix, UpperTriPartitionedMatrix) = bLU(A)

      val z = recLTriagMultiSolve(
        lmat, V, Stream(),
        V.map(c => (c._1, DenseMatrix.zeros[Double](c._2.rows, c._2.cols))))


      recUTriagMultiSolve(umat, z, Stream(), z.map(c => (c._1, DenseMatrix.zeros[Double](c._2.rows, c._2.cols))))
    }
  }


  implicit object implOpSolveUpperTriPartitionedMatrixByVector
    extends OpSolveMatrixBy.Impl2[UpperTriPartitionedMatrix, PartitionedVector, PartitionedVector] {

    override def apply(A: UpperTriPartitionedMatrix, V: PartitionedVector): PartitionedVector = {
      require(A.rows == V.rows && A.cols == V.rows, "Non-conformant matrix-vector sizes")
      require(A.colBlocks == V.rowBlocks && A.rowBlocks == A.rowBlocks, "Non-conformant matrix-vector partitions")

      recUTriagSolve(A, V, Stream(), V.map(c => (c._1, DenseVector.zeros[Double](c._2.length))))
    }
  }

  implicit object implOpSolveLowerTriPartitionedMatrixByMatrix
    extends OpSolveMatrixBy.Impl2[LowerTriPartitionedMatrix, PartitionedMatrix, PartitionedMatrix] {

    override def apply(A: LowerTriPartitionedMatrix, V: PartitionedMatrix): PartitionedMatrix = {
      require(A.rows == V.rows && A.cols == V.rows, "Non-conformant matrix-vector sizes")
      require(A.colBlocks == V.rowBlocks && A.rowBlocks == A.rowBlocks, "Non-conformant matrix-vector partitions")

      recLTriagMultiSolve(A, V, Stream(), V.map(c => (c._1, DenseMatrix.zeros[Double](c._2.rows, c._2.cols))))
    }
  }

  implicit object implOpSolveUpperTriPartitionedMatrixByMatrix
    extends OpSolveMatrixBy.Impl2[UpperTriPartitionedMatrix, PartitionedMatrix, PartitionedMatrix] {

    override def apply(A: UpperTriPartitionedMatrix, V: PartitionedMatrix): PartitionedMatrix = {
      require(A.rows == V.rows && A.cols == V.rows, "Non-conformant matrix-vector sizes")
      require(A.colBlocks == V.rowBlocks && A.rowBlocks == A.rowBlocks, "Non-conformant matrix-vector partitions")

      recUTriagMultiSolve(A, V, Stream(), V.map(c => (c._1, DenseMatrix.zeros[Double](c._2.rows, c._2.cols))))
    }
  }


}
