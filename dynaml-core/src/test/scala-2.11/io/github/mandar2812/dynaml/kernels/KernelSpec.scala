package io.github.mandar2812.dynaml.kernels

import breeze.linalg._
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.analysis.implicits._
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.pipes.DataPipe
import org.scalatest.{FlatSpec, Matchers}

class KernelSpec extends FlatSpec with Matchers {

  "Covariance Functions " should "be able to block/un-block parameters correctly" in {

    implicit val field = VectorField(1)

    val seKernel = new SEKernel(band = 1.0, h = 1.0)

    val hyp = seKernel.hyper_parameters

    seKernel.block(hyp.head)

    assert(seKernel.effective_hyper_parameters.head == hyp.last)

    seKernel.block_all_hyper_parameters

    assert(seKernel.effective_hyper_parameters.isEmpty)

    seKernel.block()

    assert(seKernel.effective_hyper_parameters.length == 2)

  }

  "RBF/SE, Cauchy, Laplace, Polynomial Kernels " should "compute correct values" in {

    val epsilon = 1E-5

    implicit val field = VectorField(1)

    val seKernel       = new SEKernel(band = 1.0, h = 1.0)

    val laplaceKernel  = new LaplacianKernel(be = 1.0)

    val polyKernel     = new PolynomialKernel(2, 1.0)

    val cauchyKernel   = new CauchyKernel(1.0)

    val (x, y, z) = (DenseVector(1.0), DenseVector(0.0), DenseVector(1.5))


    assert(math.abs(laplaceKernel.evaluate(x, y) - 0.36787944117144233) < epsilon)

    assert(math.abs(seKernel.evaluate(x, y) - 0.6065306597126334) < epsilon)

    assert(math.abs(polyKernel.evaluate(x, z) - 6.25) < epsilon)

    assert(math.abs(cauchyKernel.evaluate(x, z) - 0.8) < epsilon)
  }


  "Kernels transformations " should "compute and handle hyper-parameters correctly" in {

    val epsilon = 1E-5

    implicit val field = VectorField(1)

    val seKernel       = new SEKernel(band = 1.0, h = 1.0)
    seKernel.block("amplitude")

    val laplaceKernel  = new LaplacianKernel(be = 1.0)

    val polyKernel     = new PolynomialKernel(2, 1.0)

    val (x, y, z) = (DenseVector(1.0), DenseVector(0.0), DenseVector(1.5))

    val k1 = seKernel + polyKernel

    val k2 = laplaceKernel * polyKernel

    val k3 = seKernel + laplaceKernel

    assert(math.abs(k1.evaluate(x, z) - 7.132496902584595) < epsilon)

    assert(math.abs(k2.evaluate(x, z) - 3.790816623203959) < epsilon)

    val block_hyp2 = Seq(polyKernel.toString.split("\\.").last+"/degree")

    k2.block(block_hyp2:_*)

    assert(k2.blocked_hyper_parameters.length == 1 && k2.blocked_hyper_parameters.head == block_hyp2.head)


    assert(
      k1.blocked_hyper_parameters.length == 1 &&
        k1.blocked_hyper_parameters.head == seKernel.toString.split("\\.").last+"/amplitude")

    assert(
      k3.blocked_hyper_parameters.length == 1 &&
        k3.blocked_hyper_parameters.head == seKernel.toString.split("\\.").last+"/amplitude")

    assert(polyKernel.blocked_hyper_parameters == Seq("degree"))

  }

  "Decomposable Kernels " should "compute correctly" in {

    val epsilon = 1E-5

    implicit val field = VectorField(1)

    implicit val enc   = breezeDVSplitEncoder(1)

    val seKernel       = new SEKernel(band = 1.0, h = 1.0)

    val polyKernel     = new PolynomialKernel(2, 1.0)

    val (x, y, z) = (DenseVector(1.0, 1.0), DenseVector(0.0, 0.0), DenseVector(1.5, 1.5))

    val k1 = new DecomposableCovariance[DenseVector[Double]](seKernel, polyKernel)

    assert(math.abs(k1.evaluate(x, z) - 7.132496902584595) < epsilon)

  }

  "Decomposable Kernels " should "handle hyper-perameters in a consistent fashion" in {

    implicit val field = VectorField(1)

    implicit val enc   = breezeDVSplitEncoder(1)

    val seKernel       = new SEKernel(band = 1.0, h = 1.0)

    val polyKernel     = new PolynomialKernel(2, 1.0)

    polyKernel.block("degree")

    val (x, y, z) = (DenseVector(1.0, 1.0), DenseVector(0.0, 0.0), DenseVector(1.5, 1.5))

    val k1 = new DecomposableCovariance[DenseVector[Double]](seKernel, polyKernel)


    assert(k1.hyper_parameters.forall(
      h => h.contains(seKernel.toString.split("\\.").last) ||
        h.contains(polyKernel.toString.split("\\.").last)
    ))

    assert(
      k1.blocked_hyper_parameters.forall(
        _.contains(polyKernel.toString.split("\\.").last+"/degree")
      )
    )

  }

  "Kernel Matrices " should "be constructed correctly" in {

    val eval1 = (x: Int, y: Int) => if(x == y) 1.0 else 0.0

    val eval2 = (x: Int, y: Int) => 1/(1.0 + math.pow(x - y, 2.0))

    val nPoints = 2

    val data: Seq[Int] = 0 until nPoints

    val k1 = SVMKernel.buildSVMKernelMatrix(data, nPoints, eval1).getKernelMatrix()
    val k2 = SVMKernel.buildSVMKernelMatrix(data, nPoints, eval2).getKernelMatrix()

    assert(k1.rows == nPoints && k1.cols == nPoints && DenseMatrix.eye[Double](nPoints) == k1)

    assert(
      k2.rows == nPoints &&
        k2.cols == nPoints &&
        DenseMatrix.tabulate[Double](nPoints, nPoints)((i, j) => if(i == j) 1.0 else 0.5) == k2)

    val k3 = SVMKernel.crossKernelMatrix(data, Seq(0), eval2)

    assert(
      k3.rows == nPoints &&
        k3.cols == 1 &&
        DenseMatrix.tabulate[Double](nPoints, 1)((i, j) => if(i == j) 1.0 else 0.5) == k3)


    val k4 = SVMKernel.buildPartitionedKernelMatrix(
      data, nPoints.toLong,
      numElementsPerRowBlock = 1,
      numElementsPerColBlock = 1,
      eval2)

    assert(k4.rows == 2 && k4.cols == 2 && k4.rowBlocks == 2 && k4.colBlocks == 2)
    assert(k4._data.forall(p =>
      if(p._1._1 == p._1._2) p._2 == DenseMatrix(1.0)
      else p._2 == DenseMatrix(0.5)))

    val k5 = SVMKernel.crossPartitonedKernelMatrix(
      data, Seq(0),
      numElementsPerRowBlock = 1,
      numElementsPerColBlock = 1,
      eval2)

    assert(k5.rows == 2 && k5.cols == 1 && k5.rowBlocks == 2 && k5.colBlocks == 1)
    assert(k5._data.forall(p =>
      if(p._1._1 == p._1._2) p._2 == DenseMatrix(1.0)
      else p._2 == DenseMatrix(0.5)))

  }

  "Covariance functions constructed from Feature Maps" should " compute and compose correctly" in {

    val epsilon = 1E-5

    implicit val field = VectorField(2)

    val data = Seq(0d, math.Pi/2)

    val phi = DataPipe[Double, DenseVector[Double]](x => DenseVector(math.cos(x), math.sin(x)))

    val id = identityPipe[Double]

    val seKernel = new SEKernel(band = 1.0, h = 1.0)
    seKernel.block("amplitude")

    val id_cov = new FeatureMapCovariance[Double, Double](id)
    val cov1 = new FeatureMapCovariance[Double, DenseVector[Double]](phi)

    val cov2 = id_cov > cov1
    val cov3 = id_cov > cov1 > seKernel

    val k1 = cov1.buildKernelMatrix(data, data.length).getKernelMatrix()
    val k2 = cov2.buildKernelMatrix(data, data.length).getKernelMatrix()
    val k3 = cov3.buildKernelMatrix(data, data.length).getKernelMatrix()

    val errMat1 = DenseMatrix.eye[Double](2) - k1

    val errMat2 = DenseMatrix.eye[Double](2) - k2

    val errMat3 = DenseMatrix.tabulate[Double](2, 2)((i, j) => if(i == j) 1.0 else math.exp(-1.0)) - k3

    assert(
      k1.rows == 2 &&
        k1.cols == 2 &&
        trace(errMat1.t*errMat1) < epsilon &&
        trace(errMat2.t*errMat2) < epsilon)

    assert(cov3.blocked_hyper_parameters == Seq("amplitude") && trace(errMat3.t*errMat3) < epsilon)

  }


}
