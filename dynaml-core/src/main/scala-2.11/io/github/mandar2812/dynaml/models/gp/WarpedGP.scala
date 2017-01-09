package io.github.mandar2812.dynaml.models.gp

import io.github.mandar2812.dynaml.kernels.LocalScalarKernel

import scala.reflect.ClassTag

/**
  * Created by mandar on 02/01/2017.
  */
abstract class WarpedGP[T, I](
  cov: LocalScalarKernel[I], n: LocalScalarKernel[I],
  data: T, num: Int)(implicit ev: ClassTag[I]) extends
  AbstractGPRegressionModel[T, I](cov, n, data, num) {

}
