/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
* */
package io.github.mandar2812.dynaml.optimization

import breeze.linalg._
import io.github.mandar2812.dynaml.graph.CausalEdge

/**
 * @author mandar2812
 */
class ConjugateGradient extends RegularizedOptimizer[DenseVector[Double],
  DenseVector[Double], Double, Iterable[CausalEdge]]{

  def getRegParam = this.regParam

  /**
   * Find the optimum value of the parameters using
   * Gradient Descent.
   *
   * @param nPoints The number of data points
   * @param initialP The initial value of the parameters
   *                 as a [[DenseVector]]
   * @param ParamOutEdges An [[java.lang.Iterable]] object
   *                      having all of the out edges of the
   *                      parameter node
   *
   * @return The value of the parameters as a [[DenseVector]]
   *
   *
   * */
  override def optimize(nPoints: Long,
                        ParamOutEdges: Iterable[CausalEdge],
                        initialP: DenseVector[Double]): DenseVector[Double] = {

    val dims = initialP.length
    //Cast as problem of form A.w = b
    //A = Phi^T . Phi + I_dims*regParam
    //b = Phi^T . Y
    val (a,b): (DenseMatrix[Double], DenseVector[Double]) = ParamOutEdges.map((edge) => {
      val phi = DenseVector(edge.getPoint().getFeatureMap())
      val label = edge.getLabel().getValue()
      val phiY: DenseVector[Double] = phi * label
      (phi*phi.t, phiY)
    }).reduce((couple1, couple2) => {
      (couple1._1+couple2._1, couple1._2+couple2._2)
    })
    val smoother:DenseMatrix[Double] = DenseMatrix.eye[Double](dims)/regParam
    smoother(-1,-1) = 0.0
    val A = a + smoother

    ConjugateGradient.runCG(A, b, initialP, 0.0001, this.numIterations)
  }
}

object ConjugateGradient {
  /**
   * Solves for x in A.x = b (where A is symmetric +ve semi-definite)
   * iteratively using the Conjugate Gradient
   * algorithm.
   * */
  def runCG(A: DenseMatrix[Double],
            b: DenseVector[Double],
            x: DenseVector[Double],
            epsilon: Double,
            MAX_ITERATIONS: Int): DenseVector[Double] = {
    val residual = b - (A*x)
    val p = residual
    var count = 1.0
    var alpha = math.pow(norm(residual, 2), 2)/(p.t * (A*p))
    var beta = 0.0
    while(norm(residual, 2) >= epsilon && count <= MAX_ITERATIONS) {
      //update x
      axpy(alpha, p, x)
      //before updating residual, calculate norm (required for beta)
      val de = math.pow(norm(residual, 2), 2)
      //update residual
      axpy(-1.0*alpha, A*p, residual)
      //calculate beta
      beta = math.pow(norm(residual, 2), 2)/de
      //update p
      p :*= beta
      axpy(1.0, residual, p)
      //update alpha
      alpha = math.pow(norm(residual, 2), 2)/(p.t * (A*p))
      count += 1
    }
    x
  }

  /**
   * Solves for X in A.X = B (where A is symmetric +ve semi-definite)
   * iteratively using the Conjugate Gradient
   * algorithm.
   * */
  def runMultiCG(A: DenseMatrix[Double],
                 b: DenseMatrix[Double],
                 x: DenseMatrix[Double],
                 epsilon: Double,
                 MAX_ITERATIONS: Int): DenseMatrix[Double] = {
    val residual:DenseMatrix[Double] = b - (A*x)
    val p = residual
    var count = 1.0

    var alphaVec = DenseVector.tabulate[Double](x.cols)(i => {
      math.pow(norm(residual(::,i), 2), 2)/(p(::,i).t * (A*p(::, i)))
    })
    var alpha = DenseMatrix.tabulate[Double](x.rows, x.cols)((i,j) => {
      alphaVec(j)
    })

    var betaVec = DenseVector.zeros[Double](x.cols)
    var beta = DenseMatrix.tabulate[Double](x.rows, x.cols)((i,j) => {
      betaVec(j)
    })

    while(count <= MAX_ITERATIONS) {
      //update x
      x :+= (alpha :* p)

      //before updating residual, calculate norm (required for beta)
      val de = DenseVector.tabulate[Double](x.cols)(i => math.pow(norm(residual(::, i), 2), 2))
      //update residual
      residual :-= (alpha :* (A*p))

      //calculate beta
      betaVec = DenseVector.tabulate[Double](x.cols)(i => math.pow(norm(residual(::,i), 2), 2)/de(i))
      beta = DenseMatrix.tabulate[Double](x.rows, x.cols)((i,j) => {
        betaVec(j)
      })

      //update p
      p :*= beta
      p :+= residual

      //update alpha
      alphaVec = DenseVector.tabulate[Double](x.cols)(i => {
        math.pow(norm(residual(::,i), 2), 2)/(p(::,i).t * (A*p(::, i)))
      })
      alpha = DenseMatrix.tabulate[Double](x.rows, x.cols)((i,j) => {
        alphaVec(j)
      })

      count += 1
    }
    x
  }
}
