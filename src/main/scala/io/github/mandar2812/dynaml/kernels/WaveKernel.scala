package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{norm, DenseVector, DenseMatrix}

/**
 * @author mandar2812
 * Cauchy Kernel given by the expression
 * K(x,y) = 1 - ||x-y||**2/(||x-y||**2 + theta**2)
 */
class WaveKernel(th: Double = 1.0)
  extends SVMKernel[DenseMatrix[Double]]
  with Serializable {
  override val hyper_parameters = List("theta")

  private var theta: Double = th

  def setc(b: Double): Unit = {
    this.theta = b
  }

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double =
    if (norm(x-y,2) != 0) math.sin(norm(x-y,2)/theta)*(theta/norm(x-y,2)) else 1.0

  override def buildKernelMatrix(mappedData: List[DenseVector[Double]],
                                 length: Int): KernelMatrix[DenseMatrix[Double]] =
    SVMKernel.buildSVMKernelMatrix(mappedData, length, this.evaluate)

  override def setHyperParameters(h: Map[String, Double]) = {
    assert(hyper_parameters.forall(h contains _),
      "All hyper parameters must be contained in the arguments")
    this.theta = h("theta")
    this
  }
}