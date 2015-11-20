package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{DenseMatrix, DenseVector}

/**
 * Standard Polynomial SVM Kernel
 * of the form K(Xi,Xj) = (Xi^T * Xj + 1.0)^r
 */
class PolynomialKernel(
    private var degree: Int = 2,
    private var offset: Double = 1.0)
  extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable{

  override val hyper_parameters = List("degree")
  
  def setdegree(d: Int): Unit = {
    this.degree = d
  }

  def setoffset(o: Int): Unit = {
    this.offset = o
  }

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double =
    Math.pow((x.t * y) + this.offset, this.degree)/(Math.pow((x.t * x) + this.offset,
      this.degree.toDouble/2.0) * Math.pow((y.t * y) + this.offset,
      this.degree.toDouble/2.0))

  override def setHyperParameters(h: Map[String, Double]) = {
    assert(hyper_parameters.forall(h contains _),
      "All hyper parameters must be contained in the arguments")
    this.degree = math.ceil(h("degree")).toInt
    //this.offset = h("offset")
    this
  }
}
