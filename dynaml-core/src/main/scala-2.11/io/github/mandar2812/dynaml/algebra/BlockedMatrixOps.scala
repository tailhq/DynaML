package io.github.mandar2812.dynaml.algebra

import breeze.generic.UFunc
import breeze.linalg.operators._
import breeze.linalg.scaleAdd

/**
  * Created by mandar on 14/10/2016.
  */
object BlockedMatrixOps extends UFunc {

  /*
  * Addition operations
  * */

  /**
    * Reference implementation for adding
    * two [[BlockedMatrix]] objects.
    *
    */
  implicit object addBlockedMatAandB extends
    OpAdd.Impl2[BlockedMatrix, BlockedMatrix, BlockedMatrix] {
    def apply(a: BlockedMatrix, b: BlockedMatrix) = {
      assert(
        a.rows == b.rows && a.cols == b.cols,
        "For matrix addition A + B, their dimensions must match")

      assert(
        a.rowBlocks == b.rowBlocks && a.colBlocks == b.colBlocks,
        "For blocked matrix addition A + B, they must have equal number of blocks")

      val mat1 = a._data
      val mat2 = b._data

      new BlockedMatrix(
        mat1.join(mat2).map(c => (c._1, c._2._1 + c._2._2)),
        a.rows, a.cols, a.rowBlocks, a.colBlocks)

    }
  }

  /**
    * Reference implementation for adding
    * two [[BlockedVector]] objects.
    *
    */
  implicit object addBlockVecAandB extends
    OpAdd.Impl2[BlockedVector, BlockedVector, BlockedVector] {
    def apply(a: BlockedVector, b: BlockedVector) = {
      assert(
        a.rows == b.rows,
        "For vector addition A + B, their dimensions must match")

      assert(
        a.rowBlocks == b.rowBlocks,
        "For blocked vector addition A + B, they must have equal number of blocks")

      val mat1 = a._data
      val mat2 = b._data

      new BlockedVector(mat1.join(mat2).map(c => (c._1, c._2._1 + c._2._2)), a.rows, a.rowBlocks)

    }
  }

  implicit object inPlaceAddBlockedVec extends OpAdd.InPlaceImpl2[BlockedVector, BlockedVector] {
    override def apply(v: BlockedVector, v2: BlockedVector): Unit = {
      val inter: BlockedVector = v + v2
      v.<~(inter)
    }
  }


  /*
  * Subtraction
  * */
  implicit object subBlockedMatAandB extends
    OpSub.Impl2[BlockedMatrix, BlockedMatrix, BlockedMatrix] {
    def apply(a: BlockedMatrix, b: BlockedMatrix) = {
      assert(
        a.rows == b.rows && a.cols == b.cols,
        "For matrix addition A + B, their dimensions must match")

      assert(
        a.rowBlocks == b.rowBlocks && a.colBlocks == b.colBlocks,
        "For blocked matrix addition A + B, they must have equal number of blocks")

      val mat1 = a._data
      val mat2 = b._data

      new BlockedMatrix(
        mat1.join(mat2).map(c => (c._1, c._2._1 - c._2._2)),
        a.rows, a.cols, a.rowBlocks, a.colBlocks)

    }
  }

  implicit object subBlockVecAandB extends
    OpSub.Impl2[BlockedVector, BlockedVector, BlockedVector] {
    def apply(a: BlockedVector, b: BlockedVector) = {
      assert(
        a.rows == b.rows,
        "For vector addition A + B, their dimensions must match")

      assert(
        a.rowBlocks == b.rowBlocks,
        "For blocked vector addition A + B, they must have equal number of blocks")

      val mat1 = a._data
      val mat2 = b._data

      new BlockedVector(mat1.join(mat2).map(c => (c._1, c._2._1 - c._2._2)), a.rows, a.rowBlocks)

    }
  }

  /*
  * Multiplication
  * */

  implicit object multBlockedVecAScalar extends
    OpMulMatrix.Impl2[BlockedVector, Double, BlockedVector] {
    def apply(a: BlockedVector, b: Double) =
      new BlockedVector(a._vector.map(c => (c._1, c._2*b)), a.rows, a.rowBlocks)
  }


  implicit object multBlockedMatAandB extends
    OpMulMatrix.Impl2[BlockedMatrix, BlockedMatrix, BlockedMatrix] {
    def apply(a: BlockedMatrix, b: BlockedMatrix) = {
      assert(
        a.cols == b.rows,
        "In matrix multiplication A.B, Num_Columns(A) = Num_Rows(B)")

      assert(
        a.colBlocks == b.rowBlocks,
        "In matrix multiplication A.B, Num_Columns(A) = Num_Rows(B)")

      new BlockedMatrix(
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
    OpMulMatrix.Impl2[BlockedMatrix, BlockedVector, BlockedVector] {
    def apply(a: BlockedMatrix, b: BlockedVector) = {
      assert(
        a.cols == b.rows,
        "In matrix multiplication A.B, Num_Columns(A) = Num_Rows(B)")

      assert(
        a.colBlocks == b.rowBlocks,
        "In matrix multiplication A.B, Num_Columns(A) = Num_Rows(B)")

      new BlockedVector(
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
    OpMulInner.Impl2[BlockedVector, BlockedVector, Double] {
    def apply(a: BlockedVector, b: BlockedVector) = {
      assert(
        a.rows == b.rows,
        "In vector dot product A.B, their dimensions must match")

      a._vector.join(b._vector).map(c => c._2._1 dot c._2._2).sum()
    }
  }


  /*
  * In place update operations
  * */

  implicit object axpyBlockedVec extends scaleAdd.InPlaceImpl3[BlockedVector, Double, BlockedVector] {
    override def apply(v: BlockedVector, v2: Double, v3: BlockedVector): Unit = {
      val inter: BlockedVector = v + (v3*v2)
      v <~ inter
    }
  }

  implicit object inPlaceMultBlockedVecAScalar extends
    OpMulScalar.InPlaceImpl2[BlockedVector, Double] {
    override def apply(v: BlockedVector, v2: Double): Unit = {
      val inter: BlockedVector = v * v2
      v <~ inter
    }
  }



}
