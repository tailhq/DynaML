package io.github.mandar2812.dynaml.algebra

import breeze.generic.UFunc
import breeze.linalg.operators._
import breeze.linalg.{DenseMatrix, DenseVector, scaleAdd}
import io.github.mandar2812.dynaml.utils

/**
  * @author mandar2812 date: 17/10/2016.
  * Reference implementations for linear algebra operations
  * on partitioned vectors and matrices.
  */
object PartitionedMatrixOps extends UFunc {

  /*
  * Addition Operations
  *
  * */

  /**
    * Reference implementation for adding
    * two [[PartitionedMatrix]] objects.
    *
    */
  implicit object addPartitionedMatAandB extends
    OpAdd.Impl2[PartitionedMatrix, PartitionedMatrix, PartitionedMatrix] {
    def apply(a: PartitionedMatrix, b: PartitionedMatrix) = {
      require(
        a.rows == b.rows && a.cols == b.cols,
        "For matrix addition A + B, their dimensions must match")

      require(
        a.rowBlocks == b.rowBlocks && a.colBlocks == b.colBlocks,
        "For blocked matrix addition A + B, they must have equal number of blocks")

      val mat1 = a._data
      val mat2 = b._data

      new PartitionedMatrix(
        mat1.zip(mat2).map(c => (c._1._1, c._1._2 + c._2._2)),
        a.rows, a.cols, a.rowBlocks, a.colBlocks)

    }
  }

  /**
    * Reference implementation for adding
    * two [[PartitionedVector]] objects.
    *
    */
  implicit object addPartitionedVecAandB extends
    OpAdd.Impl2[PartitionedVector, PartitionedVector, PartitionedVector] {
    def apply(a: PartitionedVector, b: PartitionedVector) = {
      require(
        a.rows == b.rows,
        "For vector addition A + B, their dimensions must match")

      require(
        a.rowBlocks == b.rowBlocks,
        "For blocked vector addition A + B, they must have equal number of blocks")

      val mat1 = a._data
      val mat2 = b._data

      new PartitionedVector(mat1.zip(mat2).map(c => (c._1._1, c._1._2 + c._2._2)), a.rows, a.rowBlocks)

    }
  }

  /*
  * Subtraction
  * */
  implicit object subPartitionedMatAandB extends
    OpSub.Impl2[PartitionedMatrix, PartitionedMatrix, PartitionedMatrix] {
    def apply(a: PartitionedMatrix, b: PartitionedMatrix) = {
      require(
        a.rows == b.rows && a.cols == b.cols,
        "For matrix addition A + B, their dimensions must match")

      require(
        a.rowBlocks == b.rowBlocks && a.colBlocks == b.colBlocks,
        "For blocked matrix addition A + B, they must have equal number of blocks")

      val mat1 = a._data
      val mat2 = b._data

      new PartitionedMatrix(
        mat1.zip(mat2).map(c => (c._1._1, c._1._2 - c._2._2)),
        a.rows, a.cols, a.rowBlocks, a.colBlocks)

    }
  }

  implicit object subPartitionedVecAandB extends
    OpSub.Impl2[PartitionedVector, PartitionedVector, PartitionedVector] {
    def apply(a: PartitionedVector, b: PartitionedVector) = {
      require(
        a.rows == b.rows,
        "For vector addition A + B, their dimensions must match")

      require(
        a.rowBlocks == b.rowBlocks,
        "For blocked vector addition A + B, they must have equal number of blocks")

      val mat1 = a._data
      val mat2 = b._data

      new PartitionedVector(mat1.zip(mat2).map(c => (c._1._1, c._1._2 - c._2._2)), a.rows, a.rowBlocks)

    }
  }

  /*
  * Multiplication
  * */

  implicit object multPartitionedVecAScalar extends
    OpMulMatrix.Impl2[PartitionedVector, Double, PartitionedVector] {
    def apply(a: PartitionedVector, b: Double) = a.map(c => (c._1, c._2*b))
  }

  implicit object multPartitionedMatAScalar extends
    OpMulMatrix.Impl2[PartitionedMatrix, Double, PartitionedMatrix] {
    def apply(a: PartitionedMatrix, b: Double) = a.map(c => (c._1, c._2*b))
  }

  implicit object multPartitionedMatAandB extends
    OpMulMatrix.Impl2[PartitionedMatrix, PartitionedMatrix, PartitionedMatrix] {
    def apply(a: PartitionedMatrix, b: PartitionedMatrix) = {
      require(
        a.cols == b.rows,
        "In matrix multiplication A.B, Num_Columns(A) = Num_Rows(B)")

      require(
        a.colBlocks == b.rowBlocks,
        "In matrix multiplication A.B, Num_Column_Blocks(A) = Num_Row_Blocks(B)")

      new PartitionedMatrix(
        utils.combine(Seq(a._data, b._data))
          .filter(c => c.head._1._2 == c.last._1._1)
          .map(c => ((c.head._1._1, c.last._1._2), c.head._2*c.last._2))
          .groupBy(_._1)
          .toStream
          .map(c =>
            (c._1, c._2.map(_._2).reduce((a,b) => a+b))
          ), a.rows, b.cols, a.rowBlocks, b.colBlocks)
    }
  }

  implicit object multPartitionedMatAVecB extends
    OpMulMatrix.Impl2[PartitionedMatrix, PartitionedVector, PartitionedVector] {
    def apply(a: PartitionedMatrix, b: PartitionedVector) = {
      require(
        a.cols == b.rows,
        "In matrix multiplication A.B, Num_Columns(A) = Num_Rows(B)")

      require(
        a.colBlocks == b.rowBlocks,
        "In matrix multiplication A.B, Num_Column_Blocks(A) = Num_Row_Blocks(B)")

      new PartitionedVector(
        utils.combine(Seq(a._data, b._data.map(c => ((c._1, 1L), c._2.toDenseMatrix))))
          .filter(c => c.head._1._2 == c.last._1._1)
          .map(c => (c.head._1._1, c.head._2*c.last._2))
          .groupBy(_._1)
          .toStream
          .map(c =>
            (c._1, c._2.map(_._2).reduce((a,b) => a+b).toDenseVector)
          ), a.rows, a.rowBlocks)
    }
  }

  implicit object innerPartitionedVecAandB extends
    OpMulInner.Impl2[PartitionedVector, PartitionedVector, Double] {
    def apply(a: PartitionedVector, b: PartitionedVector) = {
      require(
        a.rows == b.rows,
        "In vector dot product A.B, their dimensions must match")

      a._data.zip(b._data).map(c => c._1._2 dot c._2._2).sum
    }
  }

  /**
    * Reference implementation taking outer product
    * between a [[PartitionedVector]] and [[PartitionedDualVector]] yielding
    * a [[PartitionedMatrix]].
    *
    */
  implicit object outMultPartitionedVecAandB extends
    OpMulMatrix.Impl2[PartitionedVector, PartitionedDualVector, PartitionedMatrix] {
    def apply(a: PartitionedVector, b: PartitionedDualVector) = {
      require(
        a.cols == b.rows,
        "In matrix multiplication A.B, Num_Columns(A) = Num_Rows(B)")

      new PartitionedMatrix(
        utils.combine(Seq(a._data, b.t._data)).toStream
          .map(c => (
            (c.head._1, c.last._1),
            c.head._2 * c.last._2.t)),
        a.rows, b.cols, a.rowBlocks, b.colBlocks)
    }
  }



  /*
  * In place update operations
  * */

  implicit object inPlaceAddPartitionedVec extends OpAdd.InPlaceImpl2[PartitionedVector, PartitionedVector] {
    override def apply(v: PartitionedVector, v2: PartitionedVector): Unit =
      v._data.zip(v2._data).foreach(c => c._1._2 :+= c._2._2)
  }


  implicit object axpyPartitionedVec extends scaleAdd.InPlaceImpl3[PartitionedVector, Double, PartitionedVector] {
    override def apply(v: PartitionedVector, v2: Double, v3: PartitionedVector): Unit = {
      v :+= (v3*v2)
    }
  }

  implicit object inPlaceMultPartitionedVecAScalar extends
    OpMulScalar.InPlaceImpl2[PartitionedVector, Double] {
    override def apply(v: PartitionedVector, v2: Double): Unit = {
      v._data.foreach(x => x._2 :*= v2)
    }
  }

  /*
  * Linear solve operators for partitioend matrices
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
          X(0L to 0L, 0L to 0L)._data.head._2 \ y(0L to 0L)._data.head._2 + acc.toBreezeVector
        val vectorBlocks = lAcc++ Stream(new PartitionedVector(Stream((0L, vSolved))))

        PartitionedVector.vertcat(vectorBlocks:_*)

      case _ =>
        val (l_hh, l_rh, l_rr) = (
          X(0L to 0L, 0L to 0L),
          X(1L until X.rowBlocks, 0L to 0L),
          new LowerTriPartitionedMatrix(
            X._underlyingdata
              .filter(c =>
                c._1._1 >= 1L && c._1._1 < X.rowBlocks &&
                  c._1._2 >= 1L && c._1._2 < X.colBlocks),
            num_row_blocks = X.rowBlocks - 1L,
            num_col_blocks = X.colBlocks - 1L)
          )

        val (y_h, y_r) = (y(0L to 0L), y(1L until y.rowBlocks))
        val (acc_h, acc_r) = (acc(0L to 0L), acc(1L until acc.rowBlocks))

        val vSolved: DenseVector[Double] = l_hh._data.head._2 \ y_h._data.head._2

        recLTriagSolve(
          l_rr, y_r,
          lAcc ++ Stream(new PartitionedVector(Stream((0L, vSolved))) + acc_h),
          acc_r+ l_rh*y_r)
    }

  /**
    * Tail recursive linear solve: x = L \ y
    * @param X An upper triangular matrix (transpose of matrix outputted by the [[bcholesky]] function)
    * @param y A partitioned vector
    * @param lAcc An accumulator storing the result block by block
    * @param acc A vector containing values to add into the result
    *
    * */
  def recUTriagSolve(X: UpperTriPartitionedMatrix,
                     y: PartitionedVector,
                     lAcc: Stream[PartitionedVector],
                     acc: PartitionedVector): PartitionedVector =
    X.colBlocks*X.rowBlocks match {
      case 1L =>
        val vSolved: DenseVector[Double] =
          X(0L to 0L, 0L to 0L)._data.head._2 \ y(0L to 0L)._data.head._2 + acc.toBreezeVector
        val vectorBlocks = lAcc ++ Stream(new PartitionedVector(Stream((0L, vSolved))))

        PartitionedVector.vertcat(vectorBlocks.reverse:_*)

      case _ =>
        val (l_hh, l_rh, l_rr) = (
          X(X.rowBlocks-1L to X.rowBlocks-1L, X.colBlocks-1L to X.colBlocks-1L),
          X(0L until X.rowBlocks-1L, X.colBlocks-1 to X.colBlocks-1),
          new UpperTriPartitionedMatrix(
            X._underlyingdata
              .filter(c =>
                c._1._1 >= 1L && c._1._1 < X.rowBlocks &&
                  c._1._2 >= 1L && c._1._2 < X.colBlocks),
            num_row_blocks = X.rowBlocks - 1L,
            num_col_blocks = X.colBlocks - 1L)
          )

        val (y_h, y_r) = (y(y.rowBlocks-1L to y.rowBlocks-1L), y(0L until y.rowBlocks-1L))
        val (acc_h, acc_r) = (acc(acc.rowBlocks-1L to acc.rowBlocks-1L), acc(0L until acc.rowBlocks-1))

        val vSolved: DenseVector[Double] = l_hh._data.head._2 \ y_h._data.head._2

        recUTriagSolve(
          l_rr, y_r,
          lAcc ++ Stream(new PartitionedVector(Stream((0L, vSolved))) + acc_h),
          acc_r+ l_rh*y_r)
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
        val vSolved = y_hh.map(c => (c._1, X_hh\c._2)) + acc

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
                  c._1._2 >= 1L && c._1._2 < X.colBlocks),
            num_row_blocks = X.rowBlocks - 1L,
            num_col_blocks = X.colBlocks - 1L)
          )

        val (y_h, y_r) = (
          y(0L to 0L,0L until y.colBlocks),
          y(1L until y.rowBlocks, 0L until y.colBlocks))

        val (acc_h, acc_r) = (
          acc(0L to 0L, 0L until acc.colBlocks),
          acc(1L until acc.rowBlocks, 0L until acc.colBlocks))

        val vSolved = y_h.map(c => (c._1, l_hh\c._2))

        recLTriagMultiSolve(
          l_rr, y_r,
          lAcc ++ Stream(vSolved + acc_h),
          acc_r + l_rh*y_r)
    }


  implicit object implOpSolveLowerTriPartitionedMatrixByVector
    extends OpSolveMatrixBy.Impl2[LowerTriPartitionedMatrix, PartitionedVector, PartitionedVector] {

    override def apply(A: LowerTriPartitionedMatrix, V: PartitionedVector): PartitionedVector = {
      require(A.rows == V.rows && A.cols == V.rows, "Non-conformant matrix-vector sizes")
      require(A.colBlocks == V.rowBlocks && A.rowBlocks == A.rowBlocks, "Non-conformant matrix-vector partitions")

      recLTriagSolve(A, V, Stream(), V.map(c => (c._1, DenseVector.zeros[Double](c._2.length))))
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

  implicit object implOpSolveUpperTriPartitionedMatrixByVector
    extends OpSolveMatrixBy.Impl2[UpperTriPartitionedMatrix, PartitionedVector, PartitionedVector] {

    override def apply(A: UpperTriPartitionedMatrix, V: PartitionedVector): PartitionedVector = {
      require(A.rows == V.rows && A.cols == V.rows, "Non-conformant matrix-vector sizes")
      require(A.colBlocks == V.rowBlocks && A.rowBlocks == A.rowBlocks, "Non-conformant matrix-vector partitions")

      recUTriagSolve(A, V, Stream(), V.map(c => (c._1, DenseVector.zeros[Double](c._2.length))))
    }
  }



}
