package io.github.mandar2812.dynaml.utils

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.pipes.{ReversibleScaler, Scaler}

/**
  * Created by mandar on 17/6/16.
  */
case class GaussianScaler(mean: DenseVector[Double], sigma: DenseVector[Double])
  extends ReversibleScaler[DenseVector[Double]]{
  override val i: Scaler[DenseVector[Double]] =
    Scaler((pattern: DenseVector[Double]) => (pattern :* sigma) + mean)

  override def run(data: DenseVector[Double]): DenseVector[Double] = (data-mean) :/ sigma
}
