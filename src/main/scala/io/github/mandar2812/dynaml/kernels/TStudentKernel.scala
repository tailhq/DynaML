package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{norm, DenseVector, DenseMatrix}

/**
  * Created by mandar on 23/11/15.
  */
class TStudentKernel(private var d: Double = 1.0)
  extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable {

  override val hyper_parameters = List("d")

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double = {
    val diff = x - y
    1.0/(1.0 + math.pow(norm(diff, 2), d))
  }

  def getD: Double = this.d

  override def setHyperParameters(h: Map[String, Double]) = {
    assert(hyper_parameters.forall(h contains _),
      "All hyper parameters must be contained in the arguments")
    //this.nu = h("nu")
    this.d = h("d")
    this
  }
}

class TStudentCovFunc(private var d: Double) extends LocalSVMKernel[Double] {
  override val hyper_parameters: List[String] = List("hurst")

  override def evaluate(x: Double, y: Double): Double =
    1.0/(1.0 + math.pow(math.abs(x-y), d))
}