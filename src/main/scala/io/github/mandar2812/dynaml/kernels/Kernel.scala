package io.github.mandar2812.dynaml.kernels

import breeze.linalg.DenseVector

/**
 * Defines a base class for kernels
 * defined on arbitrary objects.
 * */
abstract class AbstractKernel[T] {
  def evaluate(x: T, y: T): Double
}


/**
 * Declares a trait Kernel which would serve
 * as a base trait for all classes implementing
 * Machine Learning Kernels.
 *
 **/

trait Kernel extends AbstractKernel[DenseVector[Double]]{

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
