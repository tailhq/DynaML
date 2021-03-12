package io.github.tailhq.dynaml.kernels

import breeze.linalg.{DenseMatrix, norm, DenseVector}

/**
 * @author tailhq
 * Wave Kernel given by the expression
 * K(x,y) = sin(||x-y||<sup>2</sup>/theta) &#215;(theta/||x-y||<sup>2</sup>)
 */
class WaveKernel(th: Double = 1.0)
  extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable {
  override val hyper_parameters = List("theta")

  state = Map("theta" -> th)

  def set_theta(t: Double): Unit = {
    state += ("theta" -> t)
  }

  override def evaluateAt(
    config: Map[String, Double])(
    x: DenseVector[Double],
    y: DenseVector[Double]): Double =
    if (norm(x-y,2) != 0) math.sin(norm(x-y,2)/config("theta"))*(config("theta")/norm(x-y,2)) else 1.0

  override def gradientAt(
    config: Map[String, Double])(
    x: DenseVector[Double],
    y: DenseVector[Double]): Map[String, Double] = {

    val diff = norm(x-y, 2)
    Map("theta" -> (-1.0*math.cos(diff/config("theta")) + math.sin(diff/config("theta"))/diff))
  }

  def theta: Double = this.state("theta")

}

class WaveCovFunc(theta: Double)
  extends LocalSVMKernel[Double] {
  override val hyper_parameters: List[String] = List("theta")

  state = Map("theta" -> theta)

  override def evaluateAt(config: Map[String, Double])(x: Double, y: Double): Double =
    if (x-y != 0) math.sin(math.pow(x-y,2)/config("theta"))*(config("theta")/math.pow(x-y,2)) else 1.0

  override def gradientAt(config: Map[String, Double])(x: Double, y: Double): Map[String, Double] = {
    val diff = math.pow(x-y, 2)
    Map("theta" -> (-1.0*math.cos(diff/config("theta")) + math.sin(diff/config("theta"))/diff))
  }
}
