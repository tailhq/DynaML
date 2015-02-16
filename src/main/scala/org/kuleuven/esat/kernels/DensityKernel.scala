package org.kuleuven.esat.kernels

import breeze.linalg.DenseVector

/**
 * Abstract class which can be extended to
 * implement various Multivariate Density
 * Kernels.
 */
trait DensityKernel extends Kernel with Serializable  {
  protected val mu: Double
  protected val r: Double

  def eval(x: DenseVector[Double]):Double

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double =
    this.eval(x - y)

  protected def derivative(n: Int, x: Double): Double

  }
