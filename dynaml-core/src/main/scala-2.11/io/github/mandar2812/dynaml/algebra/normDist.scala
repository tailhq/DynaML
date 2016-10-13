package io.github.mandar2812.dynaml.algebra

import breeze.generic.UFunc

/**
  * Created by mandar on 13/10/2016.
  */
object normDist extends UFunc {

  implicit object implDV extends Impl2[SparkVector, Double, Double] {
    def apply(a: SparkVector, p: Double) = {
      assert(p >= 1.0, "For an L_p norm to be computed p >= 1.0")
      math.pow(a._vector.values.map(x => math.pow(math.abs(x), p)).sum(), 1.0/p)
    }
  }
}
