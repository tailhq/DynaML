package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{DenseMatrix, norm, DenseVector}

/**
 * Implementation of the Normalized Exponential Kernel
 *
 * K(x,y) = exp(beta*(x.y))
 */
class LaplacianKernel(be: Double = 1.0)
  extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable {
  override val hyper_parameters = List("beta")

  state = Map("beta" -> be)

  private var beta: Double = be

  def setbeta(b: Double): Unit = {
    this.beta = b
    state += ("beta" -> b)
  }

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double =
    math.exp(-1.0*beta*norm(x - y, 1)/(norm(x,1)*norm(y,1)))

}

class LaplaceCovFunc(private var beta: Double)
  extends LocalSVMKernel[Double] {
  override val hyper_parameters: List[String] = List("beta")

  state = Map("beta" -> beta)

  override def evaluate(x: Double, y: Double): Double = {
    val diff = math.abs(x - y)
    math.exp(-1.0*diff/state("beta"))
  }
}
