package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{DenseVector, DenseMatrix}

/**
  * Created by mandar on 24/11/15.
  */
class WaveletKernel(func: (Double) => Double)(private var scale: Double)
  extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable {

  override val hyper_parameters = List("scale")

  val mother: (Double) => Double = func

  def setscale(d: Double): Unit = {
    this.scale = d
  }

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double = {
    val diff = x - y
    //Math.exp(-1*math.pow(norm(diff, 2), 2)/(2*math.pow(bandwidth, 2)))
    (0 to x.length).map(i => mother(math.abs(x(i)-y(i))/scale)).product
  }

  def getscale: Double = this.scale

  override def setHyperParameters(h: Map[String, Double]) = {
    assert(hyper_parameters.forall(h contains _),
      "All hyper parameters must be contained in the arguments")
    this.scale = h("scale")
    this
  }

}

class WaveletCovFunc(func: (Double) => Double)(private var scale: Double) extends LocalSVMKernel[Double] {
  override val hyper_parameters: List[String] = List("bandwidth")

  val mother: (Double) => Double = func

  override def evaluate(x: Double, y: Double): Double = mother(math.abs(x-y)/scale)
}
