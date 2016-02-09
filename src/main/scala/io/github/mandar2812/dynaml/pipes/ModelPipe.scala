package io.github.mandar2812.dynaml.pipes

import breeze.linalg.{DenseVector, DenseMatrix}
import io.github.mandar2812.dynaml.kernels.CovarianceFunction
import io.github.mandar2812.dynaml.models._
import io.github.mandar2812.dynaml.models.gp.AbstractGPRegressionModel

/**
  * Top level trait for Pipes involving ML models.
  */
trait ModelPipe[Source, T, Q, R, M <: Model[T, Q, R]]
  extends DataPipe[Source, M]{

  val preProcess: (Source) => T

  override def run(data: Source): M
}


class GPRegressionPipe[M <:
AbstractGPRegressionModel[Seq[(DenseVector[Double], Double)],
  DenseVector[Double]], Source](pre: (Source) => Seq[(DenseVector[Double], Double)],
                                cov: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]],
                                n: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]],
                                order: Int, ex: Int)
  extends ModelPipe[Source, Seq[(DenseVector[Double], Double)],
    DenseVector[Double], Double, M] {

  override val preProcess: (Source) => Seq[(DenseVector[Double], Double)] = pre

  override def run(data: Source): M =
    AbstractGPRegressionModel[M](preProcess(data), cov, n, order, ex)

}


