package io.github.mandar2812.dynaml.models.gp

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.kernels.CovarianceFunction

/**
  * @author mandar2812
  *
  * GP-NARX
  * Gaussian Process Non-Linear
  * Auto-regressive Model with
  * Exogenous Inputs.
  *
  * y(t) = f(x(t)) + e
  * x(t) = (y(t-1), ... , y(t-p), u(t-1), ..., u(t-p))
  * f(x) ~ GP(0, cov(X,X))
  * e|f(x) ~ N(0, noise(X,X))
  */
class GPNarXModel(order: Int,
                  ex: Int,
                  cov: CovarianceFunction[DenseVector[Double],
                    Double, DenseMatrix[Double]],
                  nL: CovarianceFunction[DenseVector[Double],
                    Double, DenseMatrix[Double]],
                  trainingdata: Seq[(DenseVector[Double], Double)]) extends
GPRegression(cov, nL, trainingdata) {

  val modelOrder = order

  val exogenousInputs = ex

}