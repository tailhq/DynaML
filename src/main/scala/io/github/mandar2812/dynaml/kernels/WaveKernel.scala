package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{DenseMatrix, norm, DenseVector}

/**
 * @author mandar2812
 * Wave Kernel given by the expression
 * K(x,y) = sin(||x-y||**2/theta)*(theta/||x-y||**2)
 */
class WaveKernel(th: Double = 1.0)
  extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable {
  override val hyper_parameters = List("theta")

  state = Map("theta" -> th)

  private var theta: Double = th

  def setc(b: Double): Unit = {
    state += ("theta" -> b)
    this.theta = b
  }

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double =
    if (norm(x-y,2) != 0) math.sin(norm(x-y,2)/state("theta"))*(state("theta")/norm(x-y,2)) else 1.0

  override def gradient(x: DenseVector[Double],
                        y: DenseVector[Double]): Map[String, Double] = {
    val diff = norm(x-y, 2)
    Map("theta" -> (-1.0*math.cos(diff/state("theta")) + math.sin(diff/state("theta"))/diff))
  }

}

class WaveCovFunc(private var theta: Double)
  extends LocalSVMKernel[Double] {
  override val hyper_parameters: List[String] = List("theta")

  state = Map("theta" -> theta)

  override def evaluate(x: Double, y: Double): Double =
    if (x-y != 0) math.sin(math.pow(x-y,2)/state("theta"))*(state("theta")/math.pow(x-y,2)) else 1.0

  override def gradient(x: Double, y: Double): Map[String, Double] = {
    val diff = math.pow(x-y, 2)
    Map("theta" -> (-1.0*math.cos(diff/state("theta")) + math.sin(diff/state("theta"))/diff))
  }
}
