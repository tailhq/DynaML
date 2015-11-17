package io.github.mandar2812.dynaml.models.gp

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.kernels.CovarianceFunction

/**
  * Created by mandar on 17/11/15.
  */
class GPRegression(cov: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]],
                   trainingdata: Seq[(DenseVector[Double], Double)]) extends
AbstractGPRegressionModel[Seq[(DenseVector[Double], Double)],
  DenseVector[Double]](cov, trainingdata,
  trainingdata.length){

  /**
    * Convert from the underlying data structure to
    * Seq[(I, Y)] where I is the index set of the GP
    * and Y is the value/label type.
    **/
  override def dataAsSeq(data: Seq[(DenseVector[Double], Double)]) = data
}
