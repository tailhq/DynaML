package io.github.mandar2812.dynaml.models

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.kernels.CovarianceFunction
import io.github.mandar2812.dynaml.models.gp.AbstractGPRegressionModel

/**
  * Created by mandar on 15/6/16.
  */
class GPRegressionPipe[M <:
AbstractGPRegressionModel[Seq[(DenseVector[Double], Double)],
  DenseVector[Double]], Source](pre: (Source) => Seq[(DenseVector[Double], Double)],
                                cov: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]],
                                n: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]],
                                order: Int = 0, ex: Int = 0)
  extends ModelPipe[Source, Seq[(DenseVector[Double], Double)],
    DenseVector[Double], Double, M] {

  override val preProcess: (Source) => Seq[(DenseVector[Double], Double)] = pre

  override def run(data: Source): M =
    AbstractGPRegressionModel[M](preProcess(data), cov, n, order, ex)

}
