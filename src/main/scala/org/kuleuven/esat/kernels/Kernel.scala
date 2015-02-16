package org.kuleuven.esat.kernels

import breeze.linalg.DenseVector

/**
 * Declares a trait Kernel which would serve
 * as a base trait for all classes implementing
 * Machine Learning Kernels.
 *
 **/

trait Kernel {

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
  def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double
}
