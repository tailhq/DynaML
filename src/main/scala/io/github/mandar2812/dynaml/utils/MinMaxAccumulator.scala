package io.github.mandar2812.dynaml.utils

import org.apache.spark.AccumulatorParam

object MinMaxAccumulator extends AccumulatorParam[(Double, Double)] {
  def zero(initialValue: (Double, Double)): (Double, Double) = {
    (Double.PositiveInfinity, Double.NegativeInfinity)
  }

  def addInPlace(v1: (Double, Double), v2: (Double, Double)): (Double, Double) = {
    (math.min(v1._1,v2._1), math.max(v1._2, v2._2))
  }
}