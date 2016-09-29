package io.github.mandar2812.dynaml.algebra

import breeze.generic.UFunc
import breeze.linalg.DenseVector

/**
  * Created by mandar on 17/6/16.
  */
object square extends UFunc {
  implicit object implDouble extends Impl[Double, Double] {
    def apply(a: Double) = math.pow(a, 2.0)
  }

  implicit object implDV extends Impl[DenseVector[Double], DenseVector[Double]] {
    def apply(a: DenseVector[Double]) = a.map(x => math.pow(x, 2.0))
  }
}