package io.github.mandar2812.dynaml.utils

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.pipes.{ReversibleScaler, Scaler}

/**
  * @author mandar2812 date 17/6/16.
  *
  */
case class MinMaxScaler(min: DenseVector[Double], max: DenseVector[Double])
  extends ReversibleScaler[DenseVector[Double]] {

  override val i: Scaler[DenseVector[Double]] =
    Scaler((data: DenseVector[Double]) => (data *:* (max-min)) + min)

  override def run(data: DenseVector[Double]): DenseVector[Double] =
    (data-min) /:/ (max-min)

  def apply(r: Range): MinMaxScaler = MinMaxScaler(min(r), max(r))

  def ++(other: MinMaxScaler) =
    MinMaxScaler(
      DenseVector(this.min.toArray++other.min.toArray),
      DenseVector(this.max.toArray++other.max.toArray))
}
