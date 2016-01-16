package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{DenseMatrix, norm, DenseVector}

/**
 * Standard RBF Kernel of the form
 * K(Xi,Xj) = exp(-||Xi - Xj||**2/2*bandwidth**2)
 */

class RBFKernel(private var bandwidth: Double = 1.0)
  extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable {

  override val hyper_parameters = List("bandwidth")

  state = Map("bandwidth" -> bandwidth)

  def setbandwidth(d: Double): Unit = {
    this.state += ("bandwidth" -> d)
    this.bandwidth = d
  }

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double = {
    val diff = x - y
    Math.exp(-1*math.pow(norm(diff, 2), 2)/(2*math.pow(this.state("bandwidth"), 2)))
  }


  override def gradient(x: DenseVector[Double],
                        y: DenseVector[Double]): Map[String, Double] =
    Map("bandwidth" -> 1.0*evaluate(x,y)*math.pow(norm(x-y,2),2)/math.pow(math.abs(state("bandwidth")), 3))

  def getBandwidth: Double = this.bandwidth

}

class RBFCovFunc(private var bandwidth: Double)
  extends LocalSVMKernel[Double] {
  override val hyper_parameters: List[String] = List("bandwidth")

  state = Map("bandwidth" -> bandwidth)

  override def evaluate(x: Double, y: Double): Double = {
    val diff = x - y
    Math.exp(-1*math.pow(diff, 2)/(2*math.pow(this.state("bandwidth"), 2)))
  }

  override def gradient(x: Double, y: Double): Map[String, Double] =
    Map("bandwidth" -> evaluate(x,y)*math.pow(x-y,2)/math.pow(math.abs(state("bandwidth")), 3))
}
