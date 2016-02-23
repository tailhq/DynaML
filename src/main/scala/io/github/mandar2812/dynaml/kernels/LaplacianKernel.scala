package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{DenseMatrix, norm, DenseVector}

/**
 * Implementation of the Normalized Exponential Kernel
 *
 * K(x,y) = exp(-||x-y||/b)
 */
class LaplacianKernel(be: Double = 1.0)
  extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable {
  override val hyper_parameters = List("beta")

  state = Map("beta" -> be)

  private var beta: Double = be

  def setbeta(b: Double): Unit = {
    this.beta = b
    state += ("beta" -> b)
  }

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double =
    math.exp(-1.0*norm(x - y, 1)/state("beta"))

  override def gradient(x: DenseVector[Double],
                        y: DenseVector[Double]): Map[String, Double] =
    Map("beta" -> 1.0*evaluate(x,y)*norm(x-y, 1)/math.pow(state("beta"), 2.0))

}

class LaplaceCovFunc(private var beta: Double)
  extends LocalSVMKernel[Double] {
  override val hyper_parameters: List[String] = List("beta")

  state = Map("beta" -> beta)

  override def evaluate(x: Double, y: Double): Double = {
    val diff = math.abs(x - y)
    math.exp(-1.0*diff/state("beta"))
  }

  override def gradient(x: Double, y: Double): Map[String, Double] =
    Map("beta" -> 1.0*evaluate(x,y)*math.abs(x-y)/math.pow(state("beta"), 2))
}
