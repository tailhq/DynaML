package io.github.mandar2812.dynaml.kernels

import spire.algebra.{Field, InnerProductSpace}
import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.pipes.Encoder

/**
  * Implements the gaussian spectral mixture kernel as outlined
  * in Wilson et. al.
  *
  * The kernel is defined as the inverse fourier transform
  * of a gaussian spectral density as is shown by Bochner's theorem.
  *
  * @tparam T The domain over which the kernel is defined
  * @param center The center of the spectral power distribution.
  * @param scale The std deviation of the spectral power distribution.
  * @param enc A reversible transformation to convert the kernel's state from
  *            a [[Map]] to tuple of [[T]], implemented as an [[Encoder]]
  *
  * */
class GaussianSpectralKernel[T](
  center: T, scale: T, enc: Encoder[Map[String, Double], (T, T)])(
  implicit field: Field[T], innerProd: InnerProductSpace[T, Double]) extends
  StationaryKernel[T, Double, DenseMatrix[Double]] with
  LocalScalarKernel[T] {

  /**
    * A reversible data pipe which can convert a configuration
    * into a tuple of [[T]] containing the center and scale
    * of the underlying gaussian spectral density.
    *
    * All classes extending [[GaussianSpectralKernel]] need to implement
    * this encoding.
    * */
  val parameterEncoding: Encoder[Map[String, Double], (T, T)] = enc

  /**
    * Helper function to output the center and scale
    * */
  def getCenterAndScale(c: Map[String, Double]): (T, T) = parameterEncoding(c)

  state = parameterEncoding.i((center, scale))

  override val hyper_parameters = state.keys.toList

  override def evalAt(config: Map[String, Double])(x: T) = {
    val (m, v) = getCenterAndScale(config)
    val xscaled = field.times(x, v)

    math.cos(2*math.Pi*innerProd.dot(m, x))*math.exp(-2.0*math.Pi*math.Pi*innerProd.dot(xscaled, xscaled))
  }

  override def gradientAt(config: Map[String, Double])(x: T, y: T) = {
    val (m, v) = getCenterAndScale(config)

    val tau = field.minus(x, y)

    val tau_sq = field.times(tau, tau)

    val grad_wrt_m = innerProd.timesl(-2.0*math.Pi*math.sin(2.0*math.Pi*innerProd.dot(m, tau)), tau)
    val grad_wrt_v = innerProd.timesl(-4.0*math.Pi*math.Pi*evalAt(config)(tau), field.times(tau_sq, v))

    parameterEncoding.i((grad_wrt_m,grad_wrt_v))
  }
}

object GaussianSpectralKernel {

  def apply[T](center: T, scale: T, pEncoding: Encoder[Map[String, Double], (T, T)])(
    implicit field: Field[T], innerProd: InnerProductSpace[T, Double]): GaussianSpectralKernel[T] =
    new GaussianSpectralKernel[T](center, scale, enc = pEncoding)

  def getEncoderforBreezeDV(dim: Int) = {
    val (centerPrefixes, scalePrefixes) = (
      (0 until dim).map(n => "c_"+n),
      (0 until dim).map(n => "s_"+n))

    val forwardMap: (Map[String, Double]) => (DenseVector[Double], DenseVector[Double]) =
      (conf: Map[String, Double]) => (
      DenseVector((0 until dim).map(i => conf("c_"+i)).toArray),
      DenseVector((0 until dim).map(i => conf("s_"+i)).toArray))

    val reverseMap = (params: (DenseVector[Double], DenseVector[Double])) => {
      params._1.mapPairs((i, v) => ("c_"+i, v)).toArray.toMap ++
        params._2.mapPairs((i, v) => ("s_"+i, v)).toArray.toMap
    }

    Encoder(forwardMap, reverseMap)
  }

}