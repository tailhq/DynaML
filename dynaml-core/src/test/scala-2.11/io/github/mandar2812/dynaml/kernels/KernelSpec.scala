package io.github.mandar2812.dynaml.kernels

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.analysis.VectorField
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

    val seKernel      = new SEKernel(band = 1.0, h = 1.0)

    val laplaceKernel = new LaplacianKernel(be = 1.0)

    val polyKernel    = new PolynomialKernel(2, 1.0)

    val cauchyKernel  = new CauchyKernel(1.0)

    val (x, y, z) = (DenseVector(1.0), DenseVector(0.0), DenseVector(1.5))


    assert(math.abs(laplaceKernel.evaluate(x, y) - 0.36787944117144233) < epsilon)

    assert(math.abs(seKernel.evaluate(x, y) - 0.6065306597126334) < epsilon)

    assert(math.abs(polyKernel.evaluate(x, z) - 6.25) < epsilon)

    assert(math.abs(cauchyKernel.evaluate(x, z) - 0.8) < epsilon)
  }


  "Kernels transformations " should "compute correctly" in {

    val epsilon = 1E-5

    implicit val field = VectorField(1)

    val seKernel      = new SEKernel(band = 1.0, h = 1.0)

    val laplaceKernel = new LaplacianKernel(be = 1.0)

    val polyKernel    = new PolynomialKernel(2, 1.0)

    val cauchyKernel  = new CauchyKernel(1.0)

    val (x, y, z) = (DenseVector(1.0), DenseVector(0.0), DenseVector(1.5))

    val k1 = seKernel + polyKernel

    val k2 = laplaceKernel * polyKernel

    assert(math.abs(k1.evaluate(x, z) - 7.132496902584595) < epsilon)

    assert(math.abs(k2.evaluate(x, z) - 3.790816623203959) < epsilon)
  }



}
