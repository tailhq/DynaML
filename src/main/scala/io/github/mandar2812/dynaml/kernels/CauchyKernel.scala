package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{DenseMatrix, norm, DenseVector}

/**
 * @author mandar2812
 * Cauchy Kernel given by the expression
 * K(x,y) = 1/(1 + ||x-y||**2/sigma**2)
 */
class CauchyKernel(si: Double = 1.0)
  extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable {
  override val hyper_parameters = List("sigma")

  state = Map("sigma" -> si)

  private var sigma: Double = si

  def setsigma(b: Double): Unit = {
    this.sigma = b
    state += ("sigma" -> b)
  }

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double =
    1/(1 + math.pow(norm(x-y, 2)/state("sigma"), 2))

}

class CauchyCovFunc(private var sigma: Double)
  extends LocalSVMKernel[Double] {
  override val hyper_parameters: List[String] = List("bandwidth")

  state = Map("sigma" -> sigma)

  override def evaluate(x: Double, y: Double): Double = {
    1/(1 + math.pow((x-y)/state("sigma"), 2))
  }
}