package io.github.mandar2812.dynaml.models.gp

import io.github.mandar2812.dynaml.algebra.{PartitionedMatrix, PartitionedPSDMatrix}
import io.github.mandar2812.dynaml.kernels.{LinearPDEKernel, LocalScalarKernel, SVMKernel}
import io.github.mandar2812.dynaml.pipes.DataPipe

import scala.reflect.ClassTag

/**
  * Modified GP formulation for inference of quantities governed
  * by linear PDEs.
  * */
abstract class GPOperatorModel[T, I: ClassTag](
  cov: LinearPDEKernel[I], n: LocalScalarKernel[(I, Double)],
  observations: T, num: Int,
  meanFunc: DataPipe[(I, Double), Double] =
  DataPipe((_:(I, Double)) => 0.0)) extends
  AbstractGPRegressionModel[T, (I, Double)](
    cov, n, observations, num) {

  override val covariance = cov

  override protected def getCrossKernelMatrix[U <: Seq[(I, Double)]](test: U): PartitionedMatrix =
    SVMKernel.crossPartitonedKernelMatrix(
      trainingData, test,
      _blockSize, _blockSize,
      covariance.invOperatorKernel)

  override protected def getTestKernelMatrix[U <: Seq[(I, Double)]](test: U): PartitionedPSDMatrix =
    SVMKernel.buildPartitionedKernelMatrix(
      test, test.length.toLong,
      _blockSize, _blockSize,
      covariance.baseKernel.evaluate)


}

object GPOperatorModel {

  def apply[T, I: ClassTag](
    cov: LinearPDEKernel[I],
    noise: LocalScalarKernel[(I, Double)],
    meanFunc: DataPipe[(I, Double), Double])(
    trainingdata: T, num: Int)(
    implicit transform: DataPipe[T, Seq[((I, Double), Double)]]) = {

    val num_points = if(num > 0) num else transform(trainingdata).length

    new GPOperatorModel[T, I](cov, noise, trainingdata, num_points, meanFunc) {
      override def dataAsSeq(data: T) = transform(data)
    }

  }
}