package io.github.tailhq.dynaml.kernels

import breeze.linalg.{DenseMatrix, DenseVector, norm}
import spire.algebra.Field

/**
 * Cauchy Kernel given by the expression
 * K(x,y) = 1/(1 + ||x-y||<sup>2</sup>/&sigma;<sup>2</sup>)
 */
class CauchyKernel(si: Double = 1.0)(implicit ev: Field[DenseVector[Double]])
  extends StationaryKernel[DenseVector[Double], Double, DenseMatrix[Double]]
    with SVMKernel[DenseMatrix[Double]]
    with LocalScalarKernel[DenseVector[Double]]
  with Serializable {
  override val hyper_parameters = List("sigma")

  state = Map("sigma" -> si)

  private var sigma: Double = si

  def setsigma(b: Double): Unit = {
    this.sigma = b
    state += ("sigma" -> b)
  }

  override def evalAt(config: Map[String, Double])(x: DenseVector[Double]) = {
    1/(1 + math.pow(norm(x, 2)/config("sigma"), 2))
  }

  override def gradientAt(
    config: Map[String, Double])(
    x: DenseVector[Double],
    y: DenseVector[Double]) =
    Map("sigma" -> 2.0*math.pow(evaluateAt(config)(x,y),2)*math.pow(norm(x-y, 2), 2)/math.pow(config("sigma"), 3))

}

class CauchyCovFunc(private var sigma: Double)
  extends LocalScalarKernel[Double] {
  override val hyper_parameters: List[String] = List("sigma")

  state = Map("sigma" -> sigma)

  override def evaluateAt(config: Map[String, Double])(x: Double, y: Double): Double = {
    1/(1 + math.pow((x-y)/config("sigma"), 2))
  }

  override def gradientAt(config: Map[String, Double])(x: Double, y: Double): Map[String, Double] = {
    Map("sigma" -> 2.0*math.pow(evaluateAt(config)(x,y),2)*math.pow(x-y, 2)/math.pow(config("sigma"), 3))
  }
}

class CoRegCauchyKernel(bandwidth: Double) extends LocalScalarKernel[Int] {

  override val hyper_parameters: List[String] = List("CoRegSigma")

  state = Map("CoRegSigma" -> bandwidth)

  override def gradientAt(config: Map[String, Double])(x: Int, y: Int): Map[String, Double] =
    Map("CoRegSigma" -> 2.0*math.pow(evaluateAt(config)(x,y), 2)*math.pow(x-y, 2)/math.pow(config("CoRegSigma"), 3))

  override def evaluateAt(config: Map[String, Double])(x: Int, y: Int): Double = {
    1/(1 + math.pow(x-y, 2)/math.pow(config("CoRegSigma"), 2))
  }
}
