package io.github.mandar2812.dynaml.utils

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.pipes.{ReversibleScaler, Scaler}

/**
  * @author mandar date 30/05/2017.
  * */
case class MeanScaler(center: DenseVector[Double]) extends ReversibleScaler[DenseVector[Double]] {

  override val i = Scaler((data: DenseVector[Double]) => data + center)

  override def run(data: DenseVector[Double]) = data - center

  def apply(r: Range): MeanScaler = MeanScaler(center(r))

  def apply(n: Int): UnivariateMeanScaler = UnivariateMeanScaler(center(n))

}

case class UnivariateMeanScaler(center: Double) extends ReversibleScaler[Double] {

  override val i = Scaler((data: Double) => data + center)

  override def run(data: Double) = data - center
}