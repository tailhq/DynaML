package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{DenseMatrix, norm, DenseVector}

/**
 * Implementation of the Normalized Exponential Kernel
 *
 * K(x,y) = exp(beta*(x.y))
 */
class ExponentialKernel(be: Double = 1.0)
  extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable {
  override val hyper_parameters = List("beta")

  state = Map("beta" -> be)

  private var beta: Double = be

  def setbeta(b: Double): Unit = {
    state += ("beta" -> b)
    this.beta = b
  }

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double =
    math.exp(state("beta")*(x.t * y)/(norm(x,2)*norm(y,2)))


}
