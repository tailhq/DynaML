package io.github.mandar2812.dynaml.algebra

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.kernels.DiracKernel
import io.github.mandar2812.dynaml.algebra.DistributedMatrixOps._
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

/**
  * Created by mandar on 30/09/2016.
  */

object LinAlgebra {
  def mult(matrix: SparkMatrix, vector: SparkVector): Array[Double] = {
    val ans: SparkVector = matrix*vector
    ans._baseVector.map(_._2).collect()
  }
}


class DistributedLinearAlgebraSpec extends FlatSpec
  with Matchers
  with BeforeAndAfter {

  private val master = "local[4]"
  private val appName = "distributed-linear-algebra-test-spark"

  private var sc: SparkContext = _

  before {
    val conf = new SparkConf()
      .setMaster(master)
      .setAppName(appName)

    sc = new SparkContext(conf)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "A distributed matrix " should "have consistent multiplication with a vector" in {


    val length = 10

    val vec = new SparkVector(sc.parallelize(Seq.fill[Double](length)(1.0)).zipWithIndex().map(c => (c._2, c._1)))

    val k = new DiracKernel(0.5)

    val list = for (i <- 0L until length.toLong; j <- 0L until length.toLong) yield ((i,j),
      k.evaluate(DenseVector(i.toDouble), DenseVector(j.toDouble)))

    val mat = new SparkMatrix(sc.parallelize(list))

    assert(vec.rows == length.toLong && vec.cols == 1L, "A vector should have consistent dimensions")

    val answer = LinAlgebra.mult(mat, vec)
    assert(answer.length == length, "Multiplication A.x should have consistent dimensions")

    assert(answer.sum == 0.5*length, "L1 Norm of solution is consistent")

  }

  "A distributed square matrix " should "have consistent multiplication with a vector" in {


    val length = 10

    val vec = new SparkVector(sc.parallelize(Seq.fill[Double](length)(1.0)).zipWithIndex().map(c => (c._2, c._1)))

    val k = new DiracKernel(0.5)

    val list = for (i <- 0L until length.toLong; j <- 0L until length.toLong) yield ((i,j),
      k.evaluate(DenseVector(i.toDouble), DenseVector(j.toDouble)))

    val mat = new SparkSquareMatrix(sc.parallelize(list))

    assert(vec.rows == length.toLong && vec.cols == 1L, "A vector should have consistent dimensions")

    val answer = LinAlgebra.mult(mat, vec)
    assert(answer.length == length, "Multiplication A.x should have consistent dimensions")

    assert(answer.sum == 0.5*length, "L1 Norm of solution is consistent")

  }


}
