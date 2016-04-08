/*
 * Copyright (c) 2016. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
 * Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
 * Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
 * Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
 * Vestibulum commodo. Ut rhoncus gravida arcu.
 */

package io.github.mandar2812.dynaml.optimization

import breeze.linalg.{DenseMatrix, DenseVector, cholesky, inv}
import breeze.numerics.sqrt

/**
  * Created by mandar on 6/4/16.
  */
class LaplacePosteriorMode[I](l: Likelihood[DenseVector[Double],
  DenseVector[Double], DenseMatrix[Double],
  (DenseVector[Double], DenseVector[Double])]) extends
  RegularizedOptimizer[DenseVector[Double], I,
    Double, (DenseMatrix[Double], DenseVector[Double])]{

  val likelihood = l

  /**
    * Solve the convex optimization problem.
    */
  override def optimize(nPoints: Long,
                        ParamOutEdges: (DenseMatrix[Double], DenseVector[Double]),
                        initialP: DenseVector[Double]): DenseVector[Double] = {
    val (kMat, y) = ParamOutEdges
    var mode = initialP

    var b = DenseVector.zeros[Double](y.length)
    var a = DenseVector.zeros[Double](y.length)

    val id = DenseMatrix.eye[Double](y.length)

    (1 to this.numIterations).foreach{ iter =>
      val wMat = likelihood.hessian(y, mode) * -1.0
      val wMatsq = sqrt(wMat)

      val L = cholesky(id + wMatsq*kMat*wMatsq)
      b = wMat*mode + likelihood.gradient(y, mode)
      val buff1 = wMatsq*kMat*b
      val buff2 = inv(L)*buff1

      a = b - inv(wMatsq*L.t)*buff2
      mode = kMat*a
    }

    mode
  }
}
