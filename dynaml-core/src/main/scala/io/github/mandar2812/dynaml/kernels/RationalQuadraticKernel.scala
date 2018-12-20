package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{DenseMatrix, norm, DenseVector}
import spire.algebra.Field


/**
  * Rational Quadratic Kernel given by the expression
  * K(x,y) = (1 + ||x-y||<sup>2</sup>/2&mu;l^2^)^-0.5 &#215; &mu;^
  * */
class RationalQuadraticKernel(shape: Double = 1.0, l: Double = 1.0)(implicit ev: Field[DenseVector[Double]])
  extends StationaryKernel[DenseVector[Double], Double, DenseMatrix[Double]]
    with SVMKernel[DenseMatrix[Double]]
    with LocalScalarKernel[DenseVector[Double]]
    with Serializable {
  override val hyper_parameters = List("mu", "l")

  state = Map("mu" -> shape, "l" -> l)

  private var mu: Double = shape

  private var lambda: Double = l

  def setShape(b: Double): Unit = {
    this.mu = b
    state += ("mu" -> b)
  }

  def setScale(lam: Double): Unit = {
    this.lambda = lam
    state += ("l" -> lam)
  }

  override def evalAt(config: Map[String, Double])(x: DenseVector[Double]): Double =
    math.pow(1 + math.pow(norm(x, 2), 2)/(config("mu")*config("l")*config("l")), -0.5*(x.length+config("mu")))

  override def gradientAt(
    config: Map[String, Double])(
    x: DenseVector[Double],
    y: DenseVector[Double]): Map[String, Double] = {

    val base = 1 + math.pow(norm(x-y, 2), 2)/(config("mu")*config("l")*config("l"))
    val exponent = -0.5*(x.length+config("mu"))

    val grad_l =
      -2.0*exponent*math.pow(norm(x-y, 2), 2.0)*math.pow(base, exponent - 1.0)/(config("mu")*math.pow(config("l"), 3.0))

    val grad_mu =
      -1.0*exponent*math.pow(base, -1.0)*math.pow(norm(x-y, 2.0), 2.0)/math.pow(config("mu")*config("l"), 2.0) -
      0.5*math.log(base)

    Map(
      "l" -> grad_l,
      "mu" -> evaluateAt(config)(x, y) * grad_mu
    )
  }

}

class RationalQuadraticCovFunc(private var c: Double)
  extends LocalSVMKernel[Double] {
  override val hyper_parameters: List[String] = List("c")

  state = Map("c" -> c)

  override def evaluateAt(config: Map[String, Double])(x: Double, y: Double): Double =
    1 - math.pow(x-y, 2)/(math.pow(x-y, 2) + config("c"))

  override def gradientAt(config: Map[String, Double])(x: Double, y: Double): Map[String, Double] = {
    Map("c" ->
      2.0*math.pow(x-y, 2)*config("c")/
        math.pow(math.pow(x-y, 2) + math.pow(config("c"), 2), 2)
    )
  }
}
