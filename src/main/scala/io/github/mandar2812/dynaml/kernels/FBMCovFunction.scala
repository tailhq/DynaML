package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{norm, DenseVector, DenseMatrix}

/**
  * Created by mandar on 20/11/15.
  */
class FBMCovFunction(private var hurst: Double)
  extends LocalSVMKernel[Double] {
  override val hyper_parameters: List[String] = List("hurst")

  override def evaluate(x: Double, y: Double): Double = {
    0.5*(math.pow(math.abs(x), 2*hurst) +
      math.pow(math.abs(y), 2*hurst) -
      math.pow(math.abs(x-y), 2*hurst))
  }
}

class FBMKernel(private var hurst: Double)
  extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable {

  override val hyper_parameters = List("hurst")

  def setbandwidth(d: Double): Unit = {
    this.hurst = d
  }

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double = {
    val diff = x - y
    //Math.exp(-1*math.pow(norm(diff, 2), 2)/(2*math.pow(hurst, 2)))
    0.5*(math.pow(norm(x, 2), 2*hurst) + math.pow(norm(y, 2), 2*hurst) -
      math.pow(norm(diff, 2), 2*hurst))
  }

  def getHurst: Double = this.hurst

  override def setHyperParameters(h: Map[String, Double]) = {
    assert(hyper_parameters.forall(h contains _),
      "All hyper parameters must be contained in the arguments")
    this.hurst = h("hurst")
    this
  }

}