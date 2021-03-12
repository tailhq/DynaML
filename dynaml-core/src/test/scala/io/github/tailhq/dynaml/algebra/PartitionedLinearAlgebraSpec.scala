package io.github.tailhq.dynaml.algebra

import breeze.linalg.det
import org.scalatest.{FlatSpec, Matchers}
import io.github.tailhq.dynaml.algebra.PartitionedMatrixOps._


/**
  * Created by mandar on 19/10/2016.
  */
class PartitionedLinearAlgebraSpec extends FlatSpec with Matchers {


  "Blocked Cholesky" should "be able to factorize P.S.D matrices" in {
    val length = 1000L
    val numRowsPerBlock = 200
    val epsilon = 1E-6

    val A: PartitionedMatrix = PartitionedMatrix(
      length, length, numRowsPerBlock, numRowsPerBlock,
      (i,j) => if(i == j) 0.25*math.exp(-0.005*i) else 0.0)

    val A_ans: PartitionedMatrix = PartitionedMatrix(
      length, length, numRowsPerBlock, numRowsPerBlock,
      (i,j) => if(i == j) 0.5*math.exp(-0.0025*i) else 0.0)


    val dat = bcholesky.choleskyPAcc(
      A, 0L, Stream()
    ).sortBy(_._1)



    val L = new LowerTriPartitionedMatrix(dat, A.rows, A.cols, A.rowBlocks, A.colBlocks)

    //println(L._underlyingdata.map(c => (c._1, c._2.rows.toString +"x"+ c._2.cols.toString)).toList)

    val (l1, u1): (LowerTriPartitionedMatrix, UpperTriPartitionedMatrix) = bLU(A)
    //println(l1._underlyingdata.map(c => (c._1, c._2.rows.toString +"x"+ c._2.cols.toString)).toList)
    //println(u1._underlyingdata.map(c => (c._1, c._2.rows.toString +"x"+ c._2.cols.toString)).toList)

    val error: PartitionedMatrix = L - A_ans

    assert(det(error.toBreezeMatrix) <= epsilon)

  }


}
