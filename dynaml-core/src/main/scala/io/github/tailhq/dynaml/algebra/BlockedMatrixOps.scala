package io.github.tailhq.dynaml.algebra

import breeze.generic.UFunc
import breeze.linalg.operators._
import breeze.linalg.scaleAdd
import io.github.tailhq.dynaml.kernels

/**
  * Created by mandar on 14/10/2016.
  */
object BlockedMatrixOps extends UFunc {

  /*
  * Addition operations
  * */

  /**
    * Reference implementation for adding
    * two [[SparkBlockedMatrix]] objects.
    *
    */
  implicit object addBlockedMatAandB extends
    OpAdd.Impl2[SparkBlockedMatrix, SparkBlockedMatrix, SparkBlockedMatrix] {
    def apply(a: SparkBlockedMatrix, b: SparkBlockedMatrix) = {
      require(
        a.rows == b.rows && a.cols == b.cols,
        "For matrix addition A + B, their dimensions must match")

      require(
        a.rowBlocks == b.rowBlocks && a.colBlocks == b.colBlocks,
        "For blocked matrix addition A + B, they must have equal number of blocks")

      val mat1 = a._data
      val mat2 = b._data

      new SparkBlockedMatrix(
        mat1.join(mat2).map(c => (c._1, c._2._1 + c._2._2)),
        a.rows, a.cols, a.rowBlocks, a.colBlocks)

    }
  }

  /**
    * Reference implementation for adding
    * two [[SparkBlockedVector]] objects.
    *
    */
  implicit object addBlockVecAandB extends
    OpAdd.Impl2[SparkBlockedVector, SparkBlockedVector, SparkBlockedVector] {
    def apply(a: SparkBlockedVector, b: SparkBlockedVector) = {
      require(
        a.rows == b.rows,
        "For vector addition A + B, their dimensions must match")

      require(
        a.rowBlocks == b.rowBlocks,
        "For blocked vector addition A + B, they must have equal number of blocks")

      val mat1 = a._data
      val mat2 = b._data

      new SparkBlockedVector(mat1.zip(mat2).map(c => (c._1._1, c._1._2 + c._2._2)), a.rows, a.rowBlocks)

    }
  }

  implicit object inPlaceAddBlockedVec extends
    OpAdd.InPlaceImpl2[SparkBlockedVector, SparkBlockedVector] {
    override def apply(v: SparkBlockedVector, v2: SparkBlockedVector): Unit = {
      val inter: SparkBlockedVector = v + v2
      v.<~(inter)
    }
  }


  /*
  * Subtraction
  * */
  implicit object subBlockedMatAandB extends
    OpSub.Impl2[SparkBlockedMatrix, SparkBlockedMatrix, SparkBlockedMatrix] {
    def apply(a: SparkBlockedMatrix, b: SparkBlockedMatrix) = {
      require(
        a.rows == b.rows && a.cols == b.cols,
        "For matrix addition A + B, their dimensions must match")

      require(
        a.rowBlocks == b.rowBlocks && a.colBlocks == b.colBlocks,
        "For blocked matrix addition A + B, they must have equal number of blocks")

      val mat1 = a._data
      val mat2 = b._data

      new SparkBlockedMatrix(
        mat1.join(mat2).map(c => (c._1, c._2._1 - c._2._2)),
        a.rows, a.cols, a.rowBlocks, a.colBlocks)

    }
  }

  implicit object subBlockVecAandB extends
    OpSub.Impl2[SparkBlockedVector, SparkBlockedVector, SparkBlockedVector] {
    def apply(a: SparkBlockedVector, b: SparkBlockedVector) = {
      require(
        a.rows == b.rows,
        "For vector addition A + B, their dimensions must match")

      require(
        a.rowBlocks == b.rowBlocks,
        "For blocked vector addition A + B, they must have equal number of blocks")

      val mat1 = a._data
      val mat2 = b._data

      new SparkBlockedVector(mat1.zip(mat2).map(c => (c._1._1, c._1._2 - c._2._2)), a.rows, a.rowBlocks)

    }
  }

  /*
  * Multiplication
  * */

  implicit object multBlockedVecAScalar extends
    OpMulMatrix.Impl2[SparkBlockedVector, Double, SparkBlockedVector] {
    def apply(a: SparkBlockedVector, b: Double) =
      new SparkBlockedVector(a._vector.map(c => (c._1, c._2*b)), a.rows, a.rowBlocks)
  }


  implicit object multBlockedMatAandB extends
    OpMulMatrix.Impl2[SparkBlockedMatrix, SparkBlockedMatrix, SparkBlockedMatrix] {
    def apply(a: SparkBlockedMatrix, b: SparkBlockedMatrix) = {
      require(
        a.cols == b.rows,
        "In matrix multiplication A.B, Num_Columns(A) = Num_Rows(B)")

      require(
        a.colBlocks == b.rowBlocks,
        "In matrix multiplication A.B, Num_Column_Blocks(A) = Num_Row_Blocks(B)")

      new SparkBlockedMatrix(
        a._data.cartesian(b._data)
          .filter(c => c._1._1._2 == c._2._1._1)
          .map(c => ((c._1._1._1, c._2._1._2), c._1._2*c._2._2))
          .groupBy(_._1)
          .map(c =>
            (c._1, c._2.map(_._2).reduce((a,b) => a+b))
          ), a.rows, b.cols, a.rowBlocks, b.colBlocks)
    }
  }

  implicit object multBlockedMatAVecB extends
    OpMulMatrix.Impl2[SparkBlockedMatrix, SparkBlockedVector, SparkBlockedVector] {
    def apply(a: SparkBlockedMatrix, b: SparkBlockedVector) = {
      require(
        a.cols == b.rows,
        "In matrix multiplication A.B, Num_Columns(A) = Num_Rows(B)")

      require(
        a.colBlocks == b.rowBlocks,
        "In matrix multiplication A.B, Num_Column_Blocks(A) = Num_Row_Blocks(B)")

      new SparkBlockedVector(
        a._data.cartesian(b._data)
          .filter(c => c._1._1._2 == c._2._1)
          .map(c => (c._1._1._1, c._1._2*c._2._2))
          .groupBy(_._1)
          .map(c =>
            (c._1, c._2.map(_._2).reduce((a,b) => a+b))
          ), a.rows, a.rowBlocks)
    }
  }

  implicit object innerBlockedVecAandB extends
    OpMulInner.Impl2[SparkBlockedVector, SparkBlockedVector, Double] {
    def apply(a: SparkBlockedVector, b: SparkBlockedVector) = {
      require(
        a.rows == b.rows,
        "In vector dot product A.B, their dimensions must match")

      a._vector.join(b._vector).map(c => c._2._1 dot c._2._2).sum()
    }
  }


  /*
  * In place update operations
  * */

  implicit object axpyBlockedVec extends scaleAdd.InPlaceImpl3[SparkBlockedVector, Double, SparkBlockedVector] {
    override def apply(v: SparkBlockedVector, v2: Double, v3: SparkBlockedVector): Unit = {
      val inter: SparkBlockedVector = v + (v3*v2)
      v <~ inter
    }
  }

  implicit object inPlaceMultBlockedVecAScalar extends
    OpMulScalar.InPlaceImpl2[SparkBlockedVector, Double] {
    override def apply(v: SparkBlockedVector, v2: Double): Unit = {
      val inter: SparkBlockedVector = v * v2
      v <~ inter
    }
  }



}
