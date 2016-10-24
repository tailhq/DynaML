package io.github.mandar2812.dynaml

import breeze.generic.UFunc
import breeze.linalg.{DenseVector, diag, product, scaleAdd, sum}
import breeze.numerics._
import org.apache.spark.annotation.Experimental

/**
  * @author mandar2812 on 13/10/2016.
  *
  * The [[algebra]] object exposes numeric and linear algebra functions
  * to the end user.
  */
package object algebra {

  def axpyDist[X <: SparkMatrix, Y <: SparkMatrix](a: Double, x: X, y: Y)(
    implicit axpy: scaleAdd.InPlaceImpl3[Y, Double, X]): Unit = {
    axpy(y, a, x)
  }

  def axpyDist(a: Double, x: SparkBlockedVector, y: SparkBlockedVector)(
    implicit axpy: scaleAdd.InPlaceImpl3[SparkBlockedVector, Double, SparkBlockedVector]): Unit = {
    axpy(y, a, x)
  }

  def axpyDist(a: Double, x: PartitionedVector, y: PartitionedVector)(
    implicit axpy: scaleAdd.InPlaceImpl3[PartitionedVector, Double, PartitionedVector]): Unit = {
    axpy(y, a, x)
  }

  object bdiag extends UFunc {

    implicit object implPartitionedMatrix extends Impl[PartitionedMatrix, PartitionedVector] {
      override def apply(v: PartitionedMatrix): PartitionedVector = {
        require(v.rows == v.cols, "Matrix must be square")
        require(v.rowBlocks == v.colBlocks, "Matrix partitioning must be uniform in rows and columns")
        new PartitionedVector(v.filterBlocks(c => c._1 == c._2).map(c => (c._1._1, diag(c._2))), v.rows, v.rowBlocks)
      }
    }

    implicit object implLPartitionedMatrix extends Impl[LowerTriPartitionedMatrix, PartitionedVector] {
      override def apply(v: LowerTriPartitionedMatrix): PartitionedVector = {
        require(v.rows == v.cols, "Matrix must be square")
        require(v.rowBlocks == v.colBlocks, "Matrix partitioning must be uniform in rows and columns")
        new PartitionedVector(v.filterBlocks(c => c._1 == c._2).map(c => (c._1._1, diag(c._2))), v.rows, v.rowBlocks)
      }
    }

    implicit object implUPartitionedMatrix extends Impl[UpperTriPartitionedMatrix, PartitionedVector] {
      override def apply(v: UpperTriPartitionedMatrix): PartitionedVector = {
        require(v.rows == v.cols, "Matrix must be square")
        require(v.rowBlocks == v.colBlocks, "Matrix partitioning must be uniform in rows and columns")
        new PartitionedVector(v.filterBlocks(c => c._1 == c._2).map(c => (c._1._1, diag(c._2))), v.rows, v.rowBlocks)
      }
    }

    implicit object implPartitionedPSDMatrix extends Impl[PartitionedPSDMatrix, PartitionedVector] {
      override def apply(v: PartitionedPSDMatrix): PartitionedVector = {
        require(v.rows == v.cols, "Matrix must be square")
        require(v.rowBlocks == v.colBlocks, "Matrix partitioning must be uniform in rows and columns")
        new PartitionedVector(v.filterBlocks(c => c._1 == c._2).map(c => (c._1._1, diag(c._2))), v.rows, v.rowBlocks)
      }
    }


  }

  object blog extends UFunc {

    implicit object implPartitionedVector extends Impl[PartitionedVector, PartitionedVector] {
      override def apply(v: PartitionedVector): PartitionedVector = {
        v.map(c => (c._1, log(c._2)))
      }
    }

    implicit object implPartitionedMatrix extends Impl[PartitionedMatrix, PartitionedMatrix] {
      override def apply(v: PartitionedMatrix): PartitionedMatrix = v.map(b => (b._1, log(b._2)))
    }

    implicit object implLPartitionedMatrix
      extends Impl[LowerTriPartitionedMatrix, LowerTriPartitionedMatrix] {

      override def apply(v: LowerTriPartitionedMatrix): LowerTriPartitionedMatrix =
        new LowerTriPartitionedMatrix(
          v._underlyingdata.map(b => (b._1, log(b._2))),
          v.rows, v.cols, v.rowBlocks, v.colBlocks)
    }

    implicit object implUPartitionedMatrix
      extends Impl[UpperTriPartitionedMatrix, UpperTriPartitionedMatrix] {

      override def apply(v: UpperTriPartitionedMatrix): UpperTriPartitionedMatrix =
        new UpperTriPartitionedMatrix(
          v._underlyingdata.map(b => (b._1, log(b._2))),
          v.rows, v.cols, v.rowBlocks, v.colBlocks)
    }
  }

  object bsum extends UFunc {

    implicit object implPartitionedVetor extends Impl[PartitionedVector, Double] {
      override def apply(v: PartitionedVector): Double = {
        v._data.map(c => sum(c._2)).sum
      }
    }
  }

  object bproduct extends UFunc {

    implicit object implPartitionedVector extends Impl[PartitionedVector, Double] {
      override def apply(v: PartitionedVector): Double = {
        v._data.map(c => product(c._2)).product
      }
    }
  }

  object btrace extends UFunc {

    implicit object implPartitionedMatrix extends Impl[PartitionedMatrix, Double] {
      override def apply(v: PartitionedMatrix): Double = {
        require(v.rows == v.cols, "For trace to be computed, block matrix must be square")
        bsum(bdiag(v))
      }
    }

    implicit object implLPartitionedMatrix extends Impl[LowerTriPartitionedMatrix, Double] {
      override def apply(v: LowerTriPartitionedMatrix): Double = bsum(bdiag(v))
    }

    implicit object implUPartitionedMatrix extends Impl[UpperTriPartitionedMatrix, Double] {
      override def apply(v: UpperTriPartitionedMatrix): Double = bsum(bdiag(v))
    }
  }

  @Experimental
  object bdet extends UFunc {

    implicit object implLPartitionedMatrix extends Impl[LowerTriPartitionedMatrix, Double] {
      override def apply(v: LowerTriPartitionedMatrix): Double = bproduct(bdiag(v))
    }

    implicit object implUPartitionedMatrix extends Impl[UpperTriPartitionedMatrix, Double] {
      override def apply(v: UpperTriPartitionedMatrix): Double = bproduct(bdiag(v))
    }

    implicit object implPartitionedMatrix extends Impl[PartitionedMatrix, Double] {

      override def apply(v: PartitionedMatrix): Double = {
        require(v.rows == v.cols, "Matrix must be square for its determinant to be defined")
        require(v.rowBlocks == v.colBlocks, "Matrix partitioning must be homogeneous")
        val (ldat, udat) = bLU.LUAcc(v, 0L, Stream(), Stream())
        val L = new LowerTriPartitionedMatrix(ldat, v.rows, v.cols, v.rowBlocks, v.colBlocks)
        val U = new UpperTriPartitionedMatrix(udat, v.rows, v.cols, v.rowBlocks, v.colBlocks)
        val ans: Double = bdet.apply(L) * bdet.apply(U)
        ans
      }
    }

  }

  object square extends UFunc {
    implicit object implDouble extends Impl[Double, Double] {
      def apply(a: Double) = math.pow(a, 2.0)
    }

    implicit object implDV extends Impl[DenseVector[Double], DenseVector[Double]] {
      def apply(a: DenseVector[Double]) = a.map(x => math.pow(x, 2.0))
    }
  }


}
