package io.github.mandar2812.dynaml.models.sgp

import breeze.linalg.DenseMatrix
import io.github.mandar2812.dynaml.kernels.LocalScalarKernel
import io.github.mandar2812.dynaml.models.{ContinuousProcessModel, SecondOrderProcessModel}
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.probability.MultGaussianPRV

import scala.reflect.ClassTag

/**
  * Created by mandar on 28/02/2017.
  */
abstract class AbstractSkewGPModel[T, I: ClassTag](
  cov: LocalScalarKernel[I], n: LocalScalarKernel[I],
  data: T, num: Int, meanFunc: DataPipe[I, Double] = DataPipe((_:I) => 0.0))
  extends ContinuousProcessModel[T, I, Double, MultGaussianPRV]
    with SecondOrderProcessModel[T, I, Double, Double, DenseMatrix[Double], MultGaussianPRV] {

}
