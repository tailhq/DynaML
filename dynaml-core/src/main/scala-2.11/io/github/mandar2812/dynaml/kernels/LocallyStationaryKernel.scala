package io.github.mandar2812.dynaml.kernels

import spire.algebra.Field
import breeze.linalg.DenseMatrix

/**
  * Implementation of locally stationary kernels as defined in
  * http://jmlr.csail.mit.edu/papers/volume2/genton01a/genton01a.pdf
  *
  * K(x,y) = K1(x+y/2)&times;K2(x-y)
  *
  * @tparam I The index set or input domain over which the kernel function is evaluated.
  * @param baseKernel The kernel given by K2, it is assumed that the user inputs a valid stationary kernel
  * @param scalingFunc The non-negative scaling function K1(.)
  *
  * */
class LocallyStationaryKernel[I](baseKernel: StationaryKernel[I, Double, DenseMatrix[Double]],
                                 scalingFunc: (I) => Double)(implicit ev: Field[I])
  extends LocalScalarKernel[I] {

  state = baseKernel.state

  override val hyper_parameters: List[String] = baseKernel.hyper_parameters

  override def evaluateAt(config: Map[String, Double])(x: I, y: I): Double =
    scalingFunc(ev.div(ev.plus(x,y),ev.fromDouble(0.5)))*baseKernel.evaluateAt(config)(x,y)

  override def gradientAt(config: Map[String, Double])(x: I, y: I) =
    baseKernel.gradientAt(config)(x, y).mapValues(_*scalingFunc(ev.div(ev.plus(x,y),ev.fromDouble(0.5))))
}
