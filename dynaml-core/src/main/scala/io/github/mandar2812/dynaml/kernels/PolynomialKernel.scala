package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{DenseMatrix, DenseVector}

/**
 * Standard Polynomial SVM Kernel
 * of the form K(x,y) = (x<sup>T</sup> . y + 1.0)<sup>r</sup>
 */
class PolynomialKernel(
    private var degree: Int = 2,
    private var offset: Double = 1.0)
  extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable{

  override val hyper_parameters = List("degree", "offset")

  state = Map("degree" -> degree, "offset" -> offset)

  def setdegree(d: Int): Unit = {
    this.degree = d
    state += ("degree" -> d.toDouble)
  }

  def setoffset(o: Double): Unit = {
    this.offset = o
    state += ("offset" -> o)
  }

  override def evaluateAt(config: Map[String, Double])(
    x: DenseVector[Double],
    y: DenseVector[Double]): Double =
    math.pow((x.t * y) + config("offset"), config("degree").toInt)

  override def setHyperParameters(h: Map[String, Double]) = {
    super.setHyperParameters(h)
    if(h contains "offset")
      state += ("offset" -> math.abs(h("offset")))
    this
  }
}
