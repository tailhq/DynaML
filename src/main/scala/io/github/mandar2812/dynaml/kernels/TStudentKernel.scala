package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{norm, DenseVector, DenseMatrix}

/**
  * T-Student Kernel
  * K(x,y) = 1/(1 + ||x - y||<sup>d</sup>)
  */
class TStudentKernel(private var d: Double = 1.0)
  extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable {

  override val hyper_parameters = List("d")

  state = Map("d" -> d)

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double = {
    val diff = x - y
    1.0/(1.0 + math.pow(norm(diff, 2), state("d")))
  }

  def getD: Double = state("d")

}

class TStudentCovFunc(private var d: Double) extends LocalSVMKernel[Double] {
  override val hyper_parameters: List[String] = List("d")

  state = Map("d" -> d)

  override def evaluate(x: Double, y: Double): Double =
    1.0/(1.0 + math.pow(math.abs(x-y), state("d")))
}