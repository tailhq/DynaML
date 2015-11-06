package io.github.mandar2812.dynaml.models.gp

import io.github.mandar2812.dynaml.kernels.AbstractKernel
import io.github.mandar2812.dynaml.models.Model


/**
 * High Level description of a Gaussian Process.
 * @author mandar2812
 * @tparam I The type of the index set (i.e. Double for time series, DenseVector for GP regression)
 * @tparam Y The type of the output label
 */
abstract class GaussianProcessModel[T <: Seq[(I, Y)], I, Y] extends Model[T] {

  val KERNEL: AbstractKernel[I]

  def getPredictiveDistribution(test: T): Unit

}
