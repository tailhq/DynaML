package io.github.mandar2812.dynaml

import breeze.linalg._
import breeze.stats.distributions.{Gaussian, Uniform}
import io.github.mandar2812.dynaml.algebra.DistributedMatrixOps._
import io.github.mandar2812.dynaml.algebra._
import io.github.mandar2812.dynaml.algebra.BlockedMatrixOps._
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.kernels.{DiracKernel, RBFKernel}
import io.github.mandar2812.dynaml.models.lm.SparkGLM
import io.github.mandar2812.dynaml.optimization.ConjugateGradient
import io.github.mandar2812.dynaml.pipes.RDDPipe
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

/**
  * Created by mandar on 30/09/2016.
  */

object LinAlgebra {
  def mult(matrix: SparkMatrix, vector: SparkVector): Array[Double] = {
    val ans: SparkVector = matrix*vector
    ans._vector.map(_._2).collect()
  }
}


class ApacheSparkSpec extends FlatSpec
  with Matchers
  with BeforeAndAfter {

  private val master = "local[4]"
  private val appName = "distributed-linear-algebra-test-spark"

  private var sc: SparkContext = _

  before {
    val conf = new SparkConf()
      .setMaster(master)
      .setAppName(appName)

    val spark = SparkSession.builder.config(conf).getOrCreate()

    sc = spark.sparkContext

    sc.setLogLevel("FATAL")

  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "RDD Pipes " should " have a consistent composition" in {

    val num = 20
    val numbers = sc.parallelize(1 to num)

    val convPipe = RDDPipe((n: Int) => n.toDouble)

    val sqPipe = RDDPipe((x: Double) => x*x)

    val sqrtPipe = RDDPipe((x: Double) => math.sqrt(x))

    val resultPipe = RDDPipe((r: RDD[Double]) => r.reduce(_+_).toInt)

    val netPipeline = convPipe > sqPipe > sqrtPipe > resultPipe

    assert(netPipeline(numbers) == num*(num+1)/2)

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

  "A distributed kernel matrix " should "must be a quadratic form" in {


    val length = 100

    val nFeat = 10
    implicit val ev = VectorField(nFeat)
    val vec = new SparkVector(sc.parallelize(Seq.fill[Double](length)(1.0)).zipWithIndex().map(c => (c._2, c._1)))

    val k = new RBFKernel(1.5)

    val list = sc.parallelize(0L until length).map(l => (l, DenseVector.tabulate(nFeat)(_ => Random.nextGaussian())))

    val mat = SparkPSDMatrix(list)(k.evaluate)

    assert(vec.rows == length.toLong && vec.cols == 1L, "A vector should have consistent dimensions")

    val answer = LinAlgebra.mult(mat, vec)
    assert(answer.length == length, "Multiplication A.x should have consistent dimensions")

    assert(answer.sum >= 0.0, "x^T.K.x >= 0")

  }

  "Distributed matrices " should " concatenate in a consistent manner" in {

    val length = 100

    val nFeat = 10
    implicit val ev = VectorField(nFeat)
    val vec = new SparkVector(sc.parallelize(Seq.fill[Double](length)(1.0)).zipWithIndex().map(c => (c._2, c._1)))

    val k1 = new RBFKernel(1.5)
    val k2 = new RBFKernel(2.5)

    val list = sc.parallelize(0L until length).map(l => (l, DenseVector.tabulate(nFeat)(_ => Random.nextGaussian())))

    val mat1 = SparkPSDMatrix(list)(k1.evaluate)
    val mat2 = SparkPSDMatrix(list)(k2.evaluate)

    val res1 = SparkMatrix.vertcat(mat1, mat2)
    val res2 = SparkMatrix.horzcat(mat1, mat2)

    assert(res1.rows == mat1.rows + mat2.rows, "R = R1 + R2")
    assert(res2.cols == mat1.cols + mat2.cols, "C = C1 + C2")

  }

  "In Place operations " should "have consistent results" in {


    val length = 100

    val nFeat = 10
    implicit val ev = VectorField(nFeat)

    val vec = new SparkVector(sc.parallelize(Seq.fill[Double](length)(1.0)).zipWithIndex().map(c => (c._2, c._1)))

    val vec1 = new SparkVector(sc.parallelize(Seq.fill[Double](length)(1.0)).zipWithIndex().map(c => (c._2, c._1)))

    val list = sc.parallelize(0L until length.toLong)

    val mat = SparkSquareMatrix(list)((i,j) => if(i == j) 1.0 else 0.0)


    axpy(-1.0,vec1,vec)

    assert(vec._vector.values.map(math.abs).sum() == 0.0,
      "Multiplication x += a*y should have consistent value")

    axpy(mat*(-1.5), vec1, vec)
    assert(vec._vector.values.sum() == -1.5*length,
      "Multiplication x += A*y should have consistent value")


  }

  "Distributed Conjugate Gradient " should "be able to solve linear systems "+
    "of the form A.x = b, where A is symmetric positive definite. " in {

    val length = 10
    val list = sc.parallelize(0L until length.toLong)
    val A = SparkPSDMatrix[Long](list.map(i => (i, i)))((i,j) => if(i == j) 0.5 else 0.0)
    val b = SparkVector(list)(_ => 1.0)

    val x = new SparkVector(sc.parallelize(Seq.fill[Double](length)(2.0)).zipWithIndex().map(c => (c._2, c._1)),
      length, false)

    val epsilon = 1E-6

    val xnew = ConjugateGradient.runCG(
      A, b,
      new SparkVector(sc.parallelize(Seq.fill[Double](length)(1.0)).zipWithIndex().map(c => (c._2, c._1))),
      epsilon, 3, false, 100)

    assert(normDist(xnew-x, 1.0) <= epsilon)
  }


  "Blocked CG " should "be able to solve linear systems "+
    "of the form A.x = b, where A is symmetric positive definite. " in {

    val length = 1261
    val numRowsPerBlock = 500
    val list = sc.parallelize(0L until length.toLong)

    val A: SparkBlockedMatrix = SparkBlockedMatrix(
      SparkPSDMatrix[Long](list.map(i => (i, i)))((i,j) => if(i == j) 0.5 else 0.0),
      numRowsPerBlock, numRowsPerBlock)

    val b: SparkBlockedVector = SparkBlockedVector(SparkVector(list)(i => 1.0), numRowsPerBlock)

    val x: SparkBlockedVector =
      SparkBlockedVector(
        new SparkVector(sc.parallelize(Seq.fill[Double](length)(2.0)).zipWithIndex().map(c => (c._2, c._1)),
          length, false), numRowsPerBlock)

    val epsilon = 1E-6

    val xnew: SparkBlockedVector = ConjugateGradient.runCG(
      A, b,
      SparkBlockedVector(
        new SparkVector(sc.parallelize(Seq.fill[Double](length)(1.0)).zipWithIndex().map(c => (c._2, c._1)),
          length, false),
        numRowsPerBlock),
      epsilon, 3,
      false, 100)


    val err_norm: Double = normBDist(xnew-x, 1.0)

    assert(err_norm <= epsilon)
  }

  "An Apache Spark regression GLM" should "be able to learn parameters using "+
    "OLS given a basis function set" in {

    //Create synthetic data set of x,y values
    //x is sampled in unit hypercube, y = w.x + noise
    val noise = new Gaussian(0.0, 0.002)
    val uniH = new Uniform(0.0, 1.0)


    val numPoints:Int = 5000

    val phi = (x: DenseVector[Double]) => {
      val (x1, x2) = (x(0), x(1))
      DenseVector(math.sin(x1), math.sin(x2), math.cos(2*x1), math.cos(2*x2))
    }

    val w = DenseVector(0.5, -0.75, 1.0, -0.25)
    val wAug = DenseVector(w.toArray ++ Array(-0.8))

    val data = (1 to numPoints).map(_ => {
      val features = DenseVector.tabulate[Double](2)(_ => uniH.draw)

      val phi_feat = DenseVector(phi(features).toArray ++ Array(1.0))

      (phi(features), (wAug dot phi_feat) + noise.draw())
    }).toStream

    val (trainingData, testData) = (data.take(4000), data.takeRight(1000))

    val epsilon = 0.85

    val trainingRDD = sc.parallelize(trainingData)

    val model = new SparkGLM(trainingRDD, trainingData.length, phi)

    model.setRegParam(0.001).learn()

    val res = testData.map(p => (model.predict(p._1), p._2)).toList

    val metrics = new RegressionMetrics(res, res.length)

    assert(metrics.Rsq >= epsilon)
  }



}
