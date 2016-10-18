package io.github.mandar2812.dynaml.algebra

import breeze.generic.UFunc
import breeze.linalg.operators._
import breeze.linalg.{DenseMatrix, DenseVector, scaleAdd}
import io.github.mandar2812.dynaml.utils

/**
  * Created by mandar on 17/10/2016.
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
      assert(
        a.rows == b.rows && a.cols == b.cols,
        "For matrix addition A + B, their dimensions must match")

      assert(
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
      assert(
        a.rows == b.rows,
        "For vector addition A + B, their dimensions must match")

      assert(
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
      assert(
        a.rows == b.rows && a.cols == b.cols,
        "For matrix addition A + B, their dimensions must match")

      assert(
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
      assert(
        a.rows == b.rows,
        "For vector addition A + B, their dimensions must match")

      assert(
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
    def apply(a: PartitionedVector, b: Double) =
      new PartitionedVector(a._data.map(c => (c._1, c._2*b)), a.rows, a.rowBlocks)
  }


  implicit object multPartitionedMatAandB extends
    OpMulMatrix.Impl2[PartitionedMatrix, PartitionedMatrix, PartitionedMatrix] {
    def apply(a: PartitionedMatrix, b: PartitionedMatrix) = {
      assert(
        a.cols == b.rows,
        "In matrix multiplication A.B, Num_Columns(A) = Num_Rows(B)")

      assert(
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
      assert(
        a.cols == b.rows,
        "In matrix multiplication A.B, Num_Columns(A) = Num_Rows(B)")

      assert(
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
      assert(
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
      assert(
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


  implicit object implOpSolveLowerTriPartitionedMatrixByVector
    extends OpSolveMatrixBy.Impl2[LowerTriPartitionedMatrix, PartitionedVector, PartitionedVector] {

    override def apply(A: LowerTriPartitionedMatrix, V: PartitionedVector): PartitionedVector = {
      require(A.rows == V.rows && A.cols == V.rows, "Non-conformant matrix-vector sizes")
      require(A.colBlocks == V.rowBlocks && A.rowBlocks == A.rowBlocks, "Non-conformant matrix-vector partitions")

      recLTriagSolve(A, V, Stream(), V.map(c => (c._1, DenseVector.zeros[Double](c._2.length))))
    }
  }

  implicit object implOpSolveUpperTriPartitionedMatrixByVector
    extends OpSolveMatrixBy.Impl2[UpperTriPartitionedMatrix, PartitionedVector, PartitionedVector] {

    override def apply(A: UpperTriPartitionedMatrix, V: PartitionedVector): PartitionedVector = {
      require(A.rows == V.rows && A.cols == V.rows, "Non-conformant matrix-vector sizes")
      require(A.colBlocks == V.rowBlocks && A.rowBlocks == A.rowBlocks, "Non-conformant matrix-vector partitions")

      recLTriagSolve(A.t, V, Stream(), V.map(c => (c._1, DenseVector.zeros[Double](c._2.length))))
    }
  }



}
