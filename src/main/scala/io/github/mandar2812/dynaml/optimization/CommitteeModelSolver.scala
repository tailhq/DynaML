package io.github.mandar2812.dynaml.optimization

import breeze.linalg.{DenseMatrix, inv, DenseVector}

/**
  * Solves the optimization problem pertaining
  * to the weights of a committee model.
  */
class CommitteeModelSolver extends
RegularizedOptimizer[Int, DenseVector[Double],
  DenseVector[Double], Double,
  Stream[(DenseVector[Double], Double)]] {
  /**
    * Solve the convex optimization problem.
    *
    * min wT.C.w    such that ||w||<sub>1</sub> = 1
    */
  override def optimize(nPoints: Long,
                        ParamOutEdges: Stream[(DenseVector[Double], Double)],
                        initialP: DenseVector[Double]): DenseVector[Double] = {

    val sumMat = ParamOutEdges.map(couple => {
      val diff = couple._1 - DenseVector.fill[Double](couple._1.length)(couple._2)
      diff * diff.t
    }).reduce((mat1, mat2) => mat1+mat2)

    sumMat :/= nPoints.toDouble
    val ones = DenseVector.ones[Double](initialP.length)
    val invMat = inv(sumMat + DenseMatrix.eye[Double](initialP.length)*regParam)
    val ans: DenseVector[Double] = invMat*ones
    val Z: Double = ones dot ans
    ans/Z
  }
}
