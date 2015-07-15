package io.github.mandar2812.dynaml.utils

import breeze.linalg.DenseVector
import org.apache.spark.AccumulatorParam

object MinMaxAccumulator extends AccumulatorParam[DenseVector[Double]] {
  def zero(initialValue: DenseVector[Double]): DenseVector[Double] = {
    DenseVector(Double.PositiveInfinity, Double.NegativeInfinity)
  }

  def addInPlace(v1: DenseVector[Double], v2: DenseVector[Double]): DenseVector[Double] = {
    v1(0) = math.min(v1(0), v2(0))
    v1(1) = math.max(v1(1), v2(1))
    v1
  }
}