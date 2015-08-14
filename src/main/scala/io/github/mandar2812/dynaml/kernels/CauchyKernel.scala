package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{norm, DenseVector, DenseMatrix}

/**
 * @author mandar2812
 * Cauchy Kernel given by the expression
 * K(x,y) = 1/(1 + ||x-y||**2/sigma**2)
 */
class CauchyKernel(si: Double = 1.0)
  extends SVMKernel[DenseMatrix[Double]]
  with Serializable {
  override val hyper_parameters = List("sigma")

  private var sigma: Double = si

  def setsigma(b: Double): Unit = {
    this.sigma = b
  }

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double =
    1/(1 + math.pow(norm(x-y, 2)/sigma, 2))

  override def buildKernelMatrix(mappedData: List[DenseVector[Double]],
                                 length: Int): KernelMatrix[DenseMatrix[Double]] =
    SVMKernel.buildSVMKernelMatrix(mappedData, length, this.evaluate)

  override def setHyperParameters(h: Map[String, Double]) = {
    assert(hyper_parameters.forall(h contains _),
      "All hyper parameters must be contained in the arguments")
    this.sigma = h("sigma")
    this
  }
}