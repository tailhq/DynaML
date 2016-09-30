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

/**
  * @author mandar2812 date: 28/09/2016.
  *
  * Implicit object which contains implementation
  * of linear algebra operations with distributed
  * matrices and vectors.
  */
object DistributedMatrixOps extends UFunc {

  /**
    * Reference implementation for adding
    * two [[SparkMatrix]] objects.
    *
    */
  implicit object addMatAandB extends
    OpAdd.Impl2[SparkMatrix, SparkMatrix, SparkMatrix] {
    def apply(a: SparkMatrix, b: SparkMatrix) = {
      assert(
        a.rows == b.rows && a.cols == b.cols,
        "For matrix addition A + B, their dimensions must match")

      val mat1 = a._matrix
      val mat2 = b._matrix

      new SparkMatrix(mat1.join(mat2).map(c => (c._1, c._2._1 + c._2._2)))

    }
  }

  /**
    * Reference implementation for subtracting
    * two [[SparkMatrix]] objects.
    *
    */
  implicit object subMatAandB extends
    OpSub.Impl2[SparkMatrix, SparkMatrix, SparkMatrix] {
    def apply(a: SparkMatrix, b: SparkMatrix) = {
      assert(
        a.rows == b.rows && a.cols == b.cols,
        "For matrix addition A + B, their dimensions must match")

      val mat1 = a._matrix
      val mat2 = b._matrix

      new SparkMatrix(mat1.join(mat2).map(c => (c._1, c._2._1 - c._2._2)))

    }
  }

  /**
    * Reference implementation for adding
    * two [[SparkVector]] objects.
    *
    */
  implicit object addVecAandB extends
    OpAdd.Impl2[SparkVector, SparkVector, SparkVector] {
    def apply(a: SparkVector, b: SparkVector) = {
      assert(
        a.rows == b.rows,
        "For vector addition A + B, their dimensions must match")

      val mat1 = a._baseVector
      val mat2 = b._baseVector

      new SparkVector(mat1.join(mat2).map(c => (c._1, c._2._1 + c._2._2)))

    }
  }

  /**
    * Reference implementation for subtracting
    * two [[SparkVector]] objects.
    *
    */
  implicit object subVecAandB extends
    OpSub.Impl2[SparkVector, SparkVector, SparkVector] {
    def apply(a: SparkVector, b: SparkVector) = {
      assert(
        a.rows == b.rows,
        "For vector addition A + B, their dimensions must match")

      val mat1 = a._baseVector
      val mat2 = b._baseVector

      new SparkVector(mat1.join(mat2).map(c => (c._1, c._2._1 - c._2._2)))

    }
  }

  /**
    * Reference implementation for multiplying
    * two [[SparkMatrix]] objects.
    *
    */
  implicit object multMatAandB extends
    OpMulMatrix.Impl2[SparkMatrix, SparkMatrix, SparkMatrix] {
    def apply(a: SparkMatrix, b: SparkMatrix) = {
      assert(
        a.cols == b.rows,
        "In matrix multiplication A.B, Num_Columns(A) = Num_Rows(B)")

      new SparkMatrix(
        a._matrix.cartesian(b._matrix)
          .filter(c => c._1._1._2 == c._2._1._1)
          .map(c => ((c._1._1._1, c._2._1._2), c._1._2*c._2._2))
          .groupBy(_._1)
          .map(c =>
            (c._1, c._2.map(_._2).sum)
          ))
    }
  }

  /**
    * Reference implementation for right multiplying
    * a [[SparkVector]] to a [[SparkMatrix]].
    *
    */
  implicit object multMatAVecB extends
    OpMulMatrix.Impl2[SparkMatrix, SparkVector, SparkVector] {
    def apply(a: SparkMatrix, b: SparkVector) = {
      assert(
        a.cols == b.rows,
        "In matrix-vector multiplication A.b, Num_Columns(A) = Num_Rows(b)")

      new SparkVector(
        a._matrix.cartesian(b._matrix)
          .filter(c => c._1._1._2 == c._2._1._1)
          .map(c => ((c._1._1._1, c._2._1._2), c._1._2*c._2._2))
          .groupBy(_._1)
          .map(c =>
            (c._1._1, c._2.map(_._2).sum)
          ))
    }
  }

  /**
    * Reference implementation for left multiplying
    * a [[SparkMatrix]] by a [[DualSparkVector]].
    *
    */
  implicit object multDVecAMatB extends
    OpMulMatrix.Impl2[DualSparkVector, SparkMatrix, DualSparkVector] {
    def apply(a: DualSparkVector, b: SparkMatrix) = {
      assert(
        a.cols == b.rows,
        "In matrix multiplication A.B, Num_Columns(A) = Num_Rows(B)")

      new DualSparkVector(
        a._matrix.cartesian(b._matrix)
          .filter(c => c._1._1._2 == c._2._1._1)
          .map(c => ((c._1._1._1, c._2._1._2), c._1._2*c._2._2))
          .groupBy(_._1)
          .map(c =>
            (c._1._2, c._2.map(_._2).sum)
          ))
    }
  }

  /**
    * Reference implementation for multiplying
    * a [[SparkMatrix]] with a scalar value.
    *
    */
  implicit object multMatAScalar extends
    OpMulScalar.Impl2[SparkMatrix, Double, SparkMatrix] {
    def apply(a: SparkMatrix, b: Double) =
      new SparkMatrix(a._matrix.map(c => (c._1, c._2*b)))

  }

  /**
    * Reference implementation taking outer product
    * between a [[SparkVector]] and [[DualSparkVector]] yielding
    * a [[SparkMatrix]].
    *
    */
  implicit object outMultVecAandB extends
    OpMulMatrix.Impl2[SparkVector, DualSparkVector, SparkMatrix] {
    def apply(a: SparkVector, b: DualSparkVector) = {
      assert(
        a.cols == b.rows,
        "In matrix multiplication A.B, Num_Columns(A) = Num_Rows(B)")

      new SparkMatrix(
        a._baseVector
          .cartesian(b._baseDualVector)
          .map(c => (
            (c._1._1, c._2._1),
            c._1._2 * c._2._2))
      )
    }
  }

  /**
    * Reference implementation for multiplying
    * a [[SparkVector]] with a scalar value.
    *
    */
  implicit object multVecAScalar extends
    OpMulScalar.Impl2[SparkVector, Double, SparkVector] {
    def apply(a: SparkVector, b: Double) =
      new SparkVector(a._baseVector.map(c => (c._1, c._2*b)))
  }

  /**
    * Reference implementation for multiplying
    * a [[DualSparkVector]] with a scalar value.
    *
    */
  implicit object multDualVecAScalar extends
    OpMulScalar.Impl2[DualSparkVector, Double, DualSparkVector] {
    def apply(a: DualSparkVector, b: Double) =
      new DualSparkVector(a._baseDualVector.map(c => (c._1, c._2*b)))
  }

  /**
    * Reference implementation for inner product
    * between a [[SparkVector]] and [[DualSparkVector]]
    */
  implicit object innerVecAandB extends
    OpMulInner.Impl2[SparkVector, SparkVector, Double] {
    def apply(a: SparkVector, b: SparkVector) = {
      assert(
        a.rows == b.rows,
        "In vector dot product A.B, their dimensions must match")

      a._baseVector.join(b._baseVector).map(c => c._2._1*c._2._2).sum()
    }
  }

}
