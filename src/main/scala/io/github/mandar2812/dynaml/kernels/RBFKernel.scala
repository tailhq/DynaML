package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{norm, DenseVector}

/**
 * Standard RBF Kernel of the form
 * K(Xi,Xj) = exp(-||Xi - Xj||**2/2*bandwidth**2)
 */

class RBFKernel(private var bandwidth: Double = 1.0)
  extends LocalSVMKernel
  with Serializable {

  override val hyper_parameters = List("bandwidth")

  def setbandwidth(d: Double): Unit = {
    this.bandwidth = d
  }

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double = {
    val diff = x - y
    Math.exp(-1*math.pow(norm(diff, 2), 2)/(2*math.pow(bandwidth, 2)))
  }

  def getBandwidth: Double = this.bandwidth

  override def setHyperParameters(h: Map[String, Double]) = {
    assert(hyper_parameters.forall(h contains _),
      "All hyper parameters must be contained in the arguments")
    this.bandwidth = h("bandwidth")
    this
  }

}
