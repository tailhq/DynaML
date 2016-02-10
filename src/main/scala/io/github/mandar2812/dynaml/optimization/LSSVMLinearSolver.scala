package io.github.mandar2812.dynaml.optimization

import breeze.linalg.{DenseMatrix, inv, DenseVector}

/**
  * Created by mandar on 9/2/16.
  */
class LSSVMLinearSolver extends
RegularizedOptimizer[Int, DenseVector[Double],
  DenseVector[Double], Double,
  (DenseMatrix[Double], DenseVector[Double])] {
  /**
    * Solve the convex optimization problem.
    *
    * A = [K + I*reg]|[1]
    *      [1.t]     |[0]
    *
    * b = [y]
    *     [0]
    *
    * return inverse(A)*b
    **/
  override def optimize(nPoints: Long,
                        ParamOutEdges: (DenseMatrix[Double], DenseVector[Double]),
                        initialP: DenseVector[Double]): DenseVector[Double] = {

    val (kernelMat,labels) = ParamOutEdges
    val smoother = DenseMatrix.eye[Double](initialP.length-1)*regParam
    val ones = DenseMatrix.ones[Double](1,nPoints.toInt)
    //Construct matrix A and b block by block
    val A = DenseMatrix.horzcat(
      DenseMatrix.vertcat(kernelMat + smoother, ones),
      DenseMatrix.vertcat(ones.t, DenseMatrix(0.0))
    )

    val b = DenseVector.vertcat(labels, DenseVector(0.0))
    inv(A)*b
  }
}
