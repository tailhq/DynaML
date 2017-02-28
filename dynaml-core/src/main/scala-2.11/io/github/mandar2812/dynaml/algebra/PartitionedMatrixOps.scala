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
    * two [[PartitionedPSDMatrix]] objects.
    *
    */
  implicit object addPartitionedPSDMatAandB extends
    OpAdd.Impl2[PartitionedPSDMatrix, PartitionedPSDMatrix, PartitionedPSDMatrix] {
    def apply(a: PartitionedPSDMatrix, b: PartitionedPSDMatrix) = {
      require(
        a.rows == b.rows && a.cols == b.cols,
        "For matrix addition A + B, their dimensions must match")

      require(
        a.rowBlocks == b.rowBlocks && a.colBlocks == b.colBlocks,
        "For blocked matrix addition A + B, they must have equal number of blocks")

      val mat1 = a._data
      val mat2 = b._data

      new PartitionedPSDMatrix(
        mat1.zip(mat2).map(c => (c._1._1, c._1._2 + c._2._2)),
        a.rows, a.cols, a.rowBlocks, a.colBlocks)

    }
  }


  implicit object addPartitionedPSDMatAandMatB extends
    OpAdd.Impl2[PartitionedPSDMatrix, PartitionedMatrix, PartitionedMatrix] {
    def apply(a: PartitionedPSDMatrix, b: PartitionedMatrix) =
      addPartitionedMatAandB(a.asInstanceOf[PartitionedMatrix], b)
  }


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

  implicit object addPartitionedLMatAandUB extends
    OpSub.Impl2[LowerTriPartitionedMatrix, UpperTriPartitionedMatrix, PartitionedMatrix] {
    def apply(a: LowerTriPartitionedMatrix, b: UpperTriPartitionedMatrix) = {
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

  implicit object addPartitionedMatAandUB extends
    OpAdd.Impl2[PartitionedMatrix, UpperTriPartitionedMatrix, PartitionedMatrix] {
    def apply(a: PartitionedMatrix, b: UpperTriPartitionedMatrix) = {
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

  implicit object subPartitionedPSDMatAandMatB extends
    OpSub.Impl2[PartitionedPSDMatrix, PartitionedMatrix, PartitionedMatrix] {
    def apply(a: PartitionedPSDMatrix, b: PartitionedMatrix) =
      subPartitionedMatAandB(a.asInstanceOf[PartitionedMatrix], b)
  }

  implicit object subPartitionedLMatAandUB extends
    OpSub.Impl2[LowerTriPartitionedMatrix, UpperTriPartitionedMatrix, PartitionedMatrix] {
    def apply(a: LowerTriPartitionedMatrix, b: UpperTriPartitionedMatrix) = {
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

  implicit object multSPartitionedVecAScalar extends
    OpMulScalar.Impl2[PartitionedVector, Double, PartitionedVector] {
    def apply(a: PartitionedVector, b: Double) = a.map(c => (c._1, c._2*b))
  }

  implicit object multMPartitionedMatAScalar extends
    OpMulMatrix.Impl2[PartitionedMatrix, Double, PartitionedMatrix] {
    def apply(a: PartitionedMatrix, b: Double) = a.map(c => (c._1, c._2*b))
  }

  implicit object elemWisemultPartitionedVecAVecB extends
    OpMulScalar.Impl2[PartitionedVector, PartitionedVector, PartitionedVector] {
    def apply(a: PartitionedVector, b: PartitionedVector) = {
      require(a.rows == b.rows,
        "For element wise multiplication, partitioned vectors must be of same dimension")
      require(a.rowBlocks == b.rowBlocks,
        "For element wise multiplication, partitioned vectors must have same number of blocks")

      PartitionedVector(a._data.zip(b._data).map(c => (c._1._1, c._1._2 :* c._2._2)), a.rows)
    }
  }

  implicit object multSPartitionedMatAScalar extends
    OpMulScalar.Impl2[PartitionedMatrix, Double, PartitionedMatrix] {
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
        "In matrix-vector multiplication A.b, Num_Columns(A) = Num_Rows(B)")

      require(
        a.colBlocks == b.rowBlocks,
        "In matrix-vector multiplication A.b, Num_Column_Blocks(A) = Num_Row_Blocks(B)")

      new PartitionedVector(
        utils.combine(Seq(a._data, b._data.map(c => ((c._1, 0L), c._2.toDenseMatrix.t))))
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
  * Division
  * */

  implicit object elemWiseDivPartitionedVecAVecB extends
    OpDiv.Impl2[PartitionedVector, PartitionedVector, PartitionedVector] {
    def apply(a: PartitionedVector, b: PartitionedVector) = {
      require(a.rows == b.rows, "For element wise division, partitioned vectors must be of same dimension")
      require(a.rowBlocks == b.rowBlocks,
        "For element wise division, partitioned vectors must have same number of blocks")

      PartitionedVector(a._data.zip(b._data).map(c => (c._1._1, c._1._2 :/ c._2._2)), a.rows)
    }
  }

  implicit object elemWiseDivPartitionedMatAMatB extends
    OpDiv.Impl2[PartitionedMatrix, PartitionedMatrix, PartitionedMatrix] {
    def apply(a: PartitionedMatrix, b: PartitionedMatrix) = {
      require(a.rows == b.rows && a.cols == b.cols,
        "For element wise division, partitioned matrices must be of same dimension")

      require(a.rowBlocks == b.rowBlocks && a.colBlocks == b.colBlocks,
        "For element wise division, partitioned matrices must have same number of blocks")

      PartitionedMatrix(a._data.zip(b._data).map(c => (c._1._1, c._1._2 :/ c._2._2)), a.rows, a.cols)
    }
  }


  implicit object elemWiseModPartitionedVecAVecB extends
    OpMod.Impl2[PartitionedVector, PartitionedVector, PartitionedVector] {
    def apply(a: PartitionedVector, b: PartitionedVector) = {
      require(a.rows == b.rows, "For element wise modulo, partitioned vectors must be of same dimension")
      require(a.rowBlocks == b.rowBlocks,
        "For element wise modulo, partitioned vectors must have same number of blocks")

      PartitionedVector(a._data.zip(b._data).map(c => (c._1._1, c._1._2 :% c._2._2)), a.rows)
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


}
