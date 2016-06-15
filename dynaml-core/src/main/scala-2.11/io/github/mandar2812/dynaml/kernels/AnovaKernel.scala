package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * @author mandar2812
  * Cauchy Kernel given by the expression
  * K(x,y) = 1/(1 + ||x-y||**2/sigma**2)
  */
class AnovaKernel(si: Double = 1.0,
                  exp: Double = 4.0,
                  d: Double = 2.0)
  extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable {
  override val hyper_parameters = List("sigma", "k", "degree")

  state = Map("sigma" -> si, "k" -> exp, "degree" -> d)

  private var sigma: Double = si

  private var k = exp

  private var degree = d

  def setsigma(b: Double): Unit = {
    this.sigma = b
    state += ("sigma" -> b)
  }

  def setk(kl: Double) = {
    this.k = kl
    state += ("k" -> kl)
  }

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double = {
    x.toArray
      .zip(y.toArray)
      .map{couple =>
        math.exp(-1.0*state("degree")*state("sigma")*math.pow(math.pow(couple._1, state("k")) -
          math.pow(couple._2, state("k")),2))
      }.sum
  }


}

class AnovaCovFunc(si: Double = 1.0,
                   exp: Double = 2.0,
                   d: Double = 2.0)
  extends LocalSVMKernel[Double] {
  override val hyper_parameters = List("sigma", "k", "degree")

  state = Map("sigma" -> si, "k" -> exp, "degree" -> d)

  private var sigma: Double = si

  private var k = exp

  private var degree = d

  def setsigma(b: Double): Unit = {
    this.sigma = b
    state += ("sigma" -> b)
  }

  def setk(kl: Double) = {
    this.k = kl
    state += ("k" -> kl)
  }

  override def evaluate(x: Double, y: Double): Double =
    math.exp(-1.0*state("d")*state("sigma")*math.pow(math.pow(x, state("k")) -
      math.pow(y, state("k")),2))
}