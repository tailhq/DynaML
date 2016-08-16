package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{DenseMatrix, norm, DenseVector}

/**
 * Cauchy Kernel given by the expression
 * K(x,y) = 1/(1 + ||x-y||<sup>2</sup>/sigma<sup>2</sup>)
 */
class CauchyKernel(si: Double = 1.0)
  extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable {
  override val hyper_parameters = List("sigma")

  state = Map("sigma" -> si)

  private var sigma: Double = si

  def setsigma(b: Double): Unit = {
    this.sigma = b
    state += ("sigma" -> b)
  }

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double =
    1/(1 + math.pow(norm(x-y, 2)/state("sigma"), 2))

  override def gradient(x: DenseVector[Double], y: DenseVector[Double]): Map[String, Double] = {
    Map("sigma" -> 2.0*math.pow(evaluate(x,y),2)*math.pow(norm(x-y, 2), 2)/math.pow(state("sigma"), 3))
  }
}

class CauchyCovFunc(private var sigma: Double)
  extends LocalSVMKernel[Double] {
  override val hyper_parameters: List[String] = List("sigma")

  state = Map("sigma" -> sigma)

  override def evaluate(x: Double, y: Double): Double = {
    1/(1 + math.pow((x-y)/state("sigma"), 2))
  }

  override def gradient(x: Double, y: Double): Map[String, Double] = {
    Map("sigma" -> 2.0*math.pow(evaluate(x,y),2)*math.pow(x-y, 2)/math.pow(state("sigma"), 3))
  }
}

class CoRegCauchyKernel(bandwidth: Double) extends LocalSVMKernel[Int] {

  override val hyper_parameters: List[String] = List("coRegSigma")

  state = Map("coRegSigma" -> bandwidth)

  override def gradient(x: Int, y: Int): Map[String, Double] =
    Map("CoRegSigma" -> 2.0*math.pow(evaluate(x,y), 2)*math.pow(x-y, 2)/math.pow(state("CoRegSigma"), 3))

  override def evaluate(x: Int, y: Int): Double = {
    1/(1 + math.pow(x-y, 2)/math.pow(state("coRegSigma"), 2))
  }
}
