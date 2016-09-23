package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * @author mandar2812 date 13/09/16.
  *
  * Implementation of the Maximum Likelihood Perceptron (MLP) kernel
  */
class MLPKernel(w: Double, b: Double) extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable{

  override val hyper_parameters = List("w", "b")

  state = Map("w" -> w, "b" -> b)

  def setw(d: Double): Unit = {
    state += ("w" -> d.toDouble)
  }

  def setoffset(o: Double): Unit = {
    state += ("b" -> o)
  }

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double =
    math.asin(
      (state("w")*(x.t*y) + state("b"))/
      (math.sqrt(state("w")*(x.t*x) + state("b") + 1) * math.sqrt(state("w")*(y.t*y) + state("b") + 1)))

}
