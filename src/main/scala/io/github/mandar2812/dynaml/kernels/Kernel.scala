package io.github.mandar2812.dynaml.kernels

import breeze.linalg.DenseVector

/**
 * Defines a base class for kernels
 * defined on arbitrary objects.
 *
 * @tparam T The domain over which kernel function
 *           k(x, y) is defined. i.e. x,y belong to T
 * @tparam V The type of value returned by the kernel function
 *           k(x,y)
 * */
abstract class AbstractKernel[T, V] {
  def evaluate(x: T, y: T): V

}

/**
  * A covariance function implementation. Covariance functions are
  * central to Stochastic Process Models as well as SVMs.
  * */
trait CovarianceFunction[T, V, M] extends AbstractKernel[T, V] {
  def buildKernelMatrix[S <: Seq[T]](mappedData: S,
                                     length: Int): KernelMatrix[M]

  def buildCrossKernelMatrix[S <: Seq[T]](dataset1: S, dataset2: S): M
}

/**
 * Base Trait for Kernels defined on Euclidean Vector Spaces.
 **/

trait Kernel extends AbstractKernel[DenseVector[Double], Double]{

  /**
   * Evaluates the value of the kernel given two
   * vectorial parameters
   *
   * @param x a local Vector.
   * @param y a local Vector.
   *
   * @return the value of the Kernel function.
   *
   * */
  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double
}
