package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{DenseMatrix, DenseVector, norm}
import spire.algebra.Field

/**
 * Implementation of the Normalized Exponential Kernel
 *
 * K(x,y) = exp(-||x-y||/b)
 */
class LaplacianKernel(be: Double = 1.0)(implicit ev: Field[DenseVector[Double]])
  extends StationaryKernel[DenseVector[Double], Double, DenseMatrix[Double]]
    with SVMKernel[DenseMatrix[Double]]
    with LocalSVMKernel[DenseVector[Double]]
  with Serializable {
  override val hyper_parameters = List("beta")

  state = Map("beta" -> be)

  private var beta: Double = be

  def setbeta(b: Double): Unit = {
    this.beta = b
    state += ("beta" -> b)
  }

  override def eval(x: DenseVector[Double]): Double =
    math.exp(-1.0*norm(x, 1)/state("beta"))

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

class CoRegLaplaceKernel(bandwidth: Double) extends LocalSVMKernel[Int] {

  override val hyper_parameters: List[String] = List("coRegLB")

  state = Map("coRegLB" -> bandwidth)

  override def gradient(x: Int, y: Int): Map[String, Double] =
    Map("coRegLB" -> 1.0*evaluate(x,y)*math.abs(x-y)/math.pow(state("coRegLB"), 2))

  override def evaluate(x: Int, y: Int): Double = {
    math.exp(-1.0*math.abs(x-y)/state("coRegLB"))
  }
}
