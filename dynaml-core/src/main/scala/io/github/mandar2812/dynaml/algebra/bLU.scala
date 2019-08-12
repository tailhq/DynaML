package io.github.mandar2812.dynaml.algebra

import breeze.generic.UFunc
import io.github.mandar2812.dynaml.algebra.PartitionedMatrixOps._
import breeze.linalg.{DenseMatrix, LU, inv, lowerTriangular, upperTriangular}

/**
  * @author mandar2812 date: 20/10/2016.
  *
  * Blocked version of LU decomposition.
  * Works on square partitioned matrices.
  */
object bLU extends UFunc {

  /**
    * Calculates the LU factorization in a tail recursive manner.
    *
    * @param mat A square [[PartitionedMatrix]]
    * @param Lacc An accumulator to store blocks of the lower triangular matrix
    * @param Uacc An accumulator to store blocks of the upper triangular matrix
    *
    * @return The LU factors as streams.
    */
  def LUAcc(mat: PartitionedMatrix, nAcc: Long,
            Lacc: Stream[((Long, Long), DenseMatrix[Double])],
            Uacc: Stream[((Long, Long), DenseMatrix[Double])]):
  (Stream[((Long, Long), DenseMatrix[Double])],
    Stream[((Long, Long), DenseMatrix[Double])]) = mat.colBlocks*mat.rowBlocks match {
    case 1L =>
      val l = LU(mat._data.head._2)
      val (fL, fU) = (lowerTriangular(l.L), upperTriangular(l.U))

      (Lacc ++ Stream(((nAcc, nAcc), fL)), Uacc ++ Stream(((nAcc, nAcc), fU)))
    case _ =>
      val A_hh = mat(0L to 0L, 0L to 0L)._data.head._2
      val A_rr = mat(1L until mat.rowBlocks, 1L until mat.colBlocks)
      val (a_rh, a_hr) = (mat(1L until mat.rowBlocks, 0L to 0L), mat(0L to 0L, 1L until mat.colBlocks))

      val l = LU(A_hh)

      val (l_hh, u_hh) = (
        PartitionedMatrix(
          Stream((
            (0L, 0L),
            inv(lowerTriangular(l.L)
              .mapPairs((key, value) => if(key._1 == key._2) 1.0 else value)))
          ),
          l.L.rows, l.L.cols),
        PartitionedMatrix(Stream(((0L, 0L), inv(upperTriangular(l.U)))), l.U.rows, l.U.cols))

      val (l_rh, u_hr) = (a_rh*u_hh, l_hh*a_hr)

      LUAcc(
        A_rr - l_rh*u_hr, nAcc+1L,
        Lacc ++
          Stream(((nAcc, nAcc),
            lowerTriangular(l.L).mapPairs((key, value) => if(key._1 == key._2) 1.0 else value))) ++
          l_rh._data.map(c => ((c._1._1 + nAcc + 1L, c._1._2 + nAcc), c._2)),
        Uacc ++
          Stream(((nAcc, nAcc), upperTriangular(l.U))) ++
          u_hr._data.map(c => ((c._1._1 + nAcc, c._1._2 + nAcc + 1L), c._2))
      )
  }

  implicit object ImplBlockLU
    extends Impl[PartitionedMatrix, (LowerTriPartitionedMatrix, UpperTriPartitionedMatrix)] {
    override def apply(v: PartitionedMatrix): (LowerTriPartitionedMatrix, UpperTriPartitionedMatrix) = {

      val (datL, datU) = LUAcc(v, 0L, Stream(), Stream())

      (new LowerTriPartitionedMatrix(datL), new UpperTriPartitionedMatrix(datU))
    }
  }
}
