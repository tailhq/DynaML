package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{DenseVector, DenseMatrix}

/**
 * 18/8/15.
 */
class LinearKernel(private var offset: Double = 0.0)
  extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable{

  override val hyper_parameters = List("offset")

  state = Map("offset" -> offset)

  def setoffset(o: Int): Unit = {
    this.offset = o
  }

  override def evaluateAt(config: Map[String, Double])(
    x: DenseVector[Double],
    y: DenseVector[Double]): Double =
    (x.t * y) + config("offset")

}
