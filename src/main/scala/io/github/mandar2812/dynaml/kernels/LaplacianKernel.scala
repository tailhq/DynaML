package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{norm, DenseVector}

/**
 * Implementation of the Normalized Exponential Kernel
 *
 * K(x,y) = exp(beta*(x.y))
 */
class LaplacianKernel(be: Double = 1.0)
  extends LocalSVMKernel
  with Serializable {
  override val hyper_parameters = List("beta")

  private var beta: Double = be

  def setbeta(b: Double): Unit = {
    this.beta = b
  }

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double =
    math.exp(-1.0*beta*norm(x - y, 1)/(norm(x,1)*norm(y,1)))

  override def setHyperParameters(h: Map[String, Double]) = {
    assert(hyper_parameters.forall(h contains _),
      "All hyper parameters must be contained in the arguments")
    this.beta = h("beta")
    this
  }
}
