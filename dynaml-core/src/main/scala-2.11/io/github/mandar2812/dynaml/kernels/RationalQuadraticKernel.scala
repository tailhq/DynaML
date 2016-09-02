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
    with LocalSVMKernel[DenseVector[Double]]
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

  override def eval(x: DenseVector[Double]): Double =
    math.pow(1 + math.pow(norm(x, 2), 2)/(state("mu")*state("l")*state("l")), -0.5*(x.length+state("mu")))

  override def gradient(x: DenseVector[Double], y: DenseVector[Double]): Map[String, Double] = {

    val base = 1 + math.pow(norm(x-y, 2), 2)/(state("mu")*state("l")*state("l"))
    val exponent = -0.5*(x.length+state("mu"))

    Map(
      "l" ->
        -2.0*exponent*math.pow(norm(x-y, 2), 2.0)*math.pow(base, exponent - 1.0)/(state("mu")*math.pow(state("l"), 3.0)),
      "mu" -> this.evaluate(x, y) * (
        -1.0*exponent*math.pow(base, -1.0)*math.pow(norm(x-y, 2.0), 2.0)/math.pow(state("mu")*state("l"), 2.0) -
        0.5*math.log(base)
        )
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
