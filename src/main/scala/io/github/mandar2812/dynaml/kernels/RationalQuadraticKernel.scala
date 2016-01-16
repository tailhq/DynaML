package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{DenseMatrix, norm, DenseVector}

/**
 * @author mandar2812
 * Cauchy Kernel given by the expression
 * K(x,y) = 1 - ||x-y||**2/(||x-y||**2 + c**2)
 */
class RationalQuadraticKernel(si: Double = 1.0)
  extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable {
  override val hyper_parameters = List("c")

  state = Map("c" -> si)

  private var c: Double = si

  def setc(b: Double): Unit = {
    this.c = b
    state += ("c" -> b)
  }

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double =
    1 - math.pow(norm(x-y, 2), 2)/(math.pow(norm(x-y, 2), 2) + state("c"))

  override def gradient(x: DenseVector[Double], y: DenseVector[Double]): Map[String, Double] = {
    Map("c" ->
      2.0*math.pow(norm(x-y, 2), 2)*state("c")/
        math.pow(math.pow(norm(x-y, 2), 2) + math.pow(state("c"), 2), 2)
    )
  }

}

class RationalQuadraticCovFunc(private var c: Double)
  extends LocalSVMKernel[Double] {
  override val hyper_parameters: List[String] = List("c")

  state = Map("c" -> c)

  override def evaluate(x: Double, y: Double): Double =
    1 - math.pow(x-y, 2)/(math.pow(x-y, 2) + state("c"))

  override def gradient(x: Double, y: Double): Map[String, Double] = {
    Map("c" ->
      2.0*math.pow(x-y, 2)*state("c")/
        math.pow(math.pow(x-y, 2) + math.pow(state("c"), 2), 2)
    )
  }
}
