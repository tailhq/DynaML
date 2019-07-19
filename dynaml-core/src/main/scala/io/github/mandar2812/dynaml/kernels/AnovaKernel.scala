package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * @author mandar2812
  * Annova Kernel
  */
class AnovaKernel(si: Double = 1.0, exp: Double = 4.0, d: Double = 2.0)
    extends SVMKernel[DenseMatrix[Double]]
    with LocalSVMKernel[DenseVector[Double]]
    with Serializable {
  override val hyper_parameters = List("sigma", "k", "degree")

  state = Map("sigma" -> si, "k" -> exp, "degree" -> d)

  def setsigma(b: Double): Unit = {
    state += ("sigma" -> b)
  }

  def setk(kl: Double) = {
    state += ("k" -> kl)
  }

  override def evaluateAt(
    config: Map[String, Double]
  )(x: DenseVector[Double],
    y: DenseVector[Double]
  ): Double = {
    x.toArray
      .zip(y.toArray)
      .map { couple =>
        math.exp(
          -1.0 * config("degree") * config("sigma") * math.pow(
            math.pow(couple._1, config("k")) -
              math.pow(couple._2, config("k")),
            2
          )
        )
      }
      .sum
  }
}

class AnovaCovFunc(si: Double = 1.0, exp: Double = 2.0, d: Double = 2.0)
    extends LocalSVMKernel[Double] {
  override val hyper_parameters = List("sigma", "k", "degree")

  state = Map("sigma" -> si, "k" -> exp, "degree" -> d)

  def setsigma(b: Double): Unit = {
    state += ("sigma" -> b)
  }

  def setk(kl: Double) = {
    state += ("k" -> kl)
  }

  override def evaluateAt(
    config: Map[String, Double]
  )(x: Double,
    y: Double
  ): Double =
    math.exp(
      -1.0 * config("d") * config("sigma") * math.pow(
        math.pow(x, config("k")) -
          math.pow(y, config("k")),
        2
      )
    )
}
