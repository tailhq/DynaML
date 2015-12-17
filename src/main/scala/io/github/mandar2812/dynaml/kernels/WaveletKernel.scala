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
    state += ("scale" -> d)
  }

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double = {
    val diff = x - y
    diff.map(i => mother(math.abs(i)/scale)).toArray.product
  }

  def getscale: Double = state("scale")


}

class WaveletCovFunc(func: (Double) => Double)(private var scale: Double)
  extends LocalSVMKernel[Double] {
  override val hyper_parameters: List[String] = List("scale")

  val mother: (Double) => Double = func

  state = Map("scale" -> scale)

  override def evaluate(x: Double, y: Double): Double = mother(math.abs(x-y)/state("scale"))
}
