package io.github.mandar2812.dynaml.models

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.kernels.CovarianceFunction
import io.github.mandar2812.dynaml.models.svm.DLSSVM

/**
  * Created by mandar on 15/6/16.
  */
class DLSSVMPipe[Source](pre: (Source) => Stream[(DenseVector[Double], Double)],
                         cov: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]],
                         task: String = "regression") extends
  ModelPipe[Source, Stream[(DenseVector[Double], Double)],
    DenseVector[Double], Double, DLSSVM] {

  override val preProcess = pre

  override def run(data: Source) = {
    val training = preProcess(data)
    new DLSSVM(training, training.length, cov, task)
  }
}
