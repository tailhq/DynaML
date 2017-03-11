package io.github.mandar2812.dynaml.kernels

import spire.algebra.{Field, InnerProductSpace}
import breeze.linalg.DenseMatrix
import io.github.mandar2812.dynaml.pipes.Encoder

/**
  * Implements the gaussian spectral kernel as outlined
  * in Wilson et. al.
  *
  * The kernel is defined as the inverse fourier transform
  * of a gaussian spectral density as is shown by Bochner's theorem.
  *
  * @tparam I The domain over which the kernel is defined
  * @param center The center of the spectral power distribution.
  * @param scale The std deviation of the spectral power distribution.
  *
  * */
class GaussianSpectralKernel[I](center: I, scale: I, enc: Encoder[Map[String, Double], (I, I)])(
  implicit field: Field[I] with InnerProductSpace[I, Double]) extends
  StationaryKernel[I, Double, DenseMatrix[Double]] with
  LocalScalarKernel[I] {

  /**
    * A reversible data pipe which can convert a configuration
    * into a tuple of [[I]] containing the center and scale
    * of the underlying gaussian spectral density.
    *
    * All classes extending [[GaussianSpectralKernel]] need to implement
    * this encoding.
    * */
  val parameterEncoding: Encoder[Map[String, Double], (I, I)] = enc

  /**
    * Helper function to output the center and scale
    * */
  def getCenterAndScale(c: Map[String, Double]): (I, I) = parameterEncoding(c)

  state = parameterEncoding.i((center, scale))

  override val hyper_parameters = state.keys.toList

  override def evalAt(config: Map[String, Double])(x: I) = {
    val (m, v) = getCenterAndScale(config)
    val xscaled = field.times(x, v)

    math.cos(2*math.Pi*field.dot(m, x))*math.exp(-2.0*math.Pi*math.Pi*field.dot(xscaled, xscaled))
  }

  override def gradientAt(config: Map[String, Double])(x: I, y: I) = {
    val (m, v) = getCenterAndScale(config)

    val tau = field.minus(x, y)

    val tau_sq = field.times(tau, tau)

    val grad_wrt_m = field.timesl(-2.0*math.Pi*math.sin(2.0*math.Pi*field.dot(m, tau)), tau)
    val grad_wrt_v = field.timesl(-4.0*math.Pi*math.Pi*evalAt(config)(tau), field.times(tau_sq, v))

    parameterEncoding.i((grad_wrt_m,grad_wrt_v))
  }
}

object GaussianSpectralKernel {

  def apply[T](center: T, scale: T, pEncoding: Encoder[Map[String, Double], (T, T)])(
    implicit field: Field[T] with InnerProductSpace[T, Double]): GaussianSpectralKernel[T] =
    new GaussianSpectralKernel[T](center, scale, enc = pEncoding)

}