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

import breeze.linalg.{DenseMatrix, inv, DenseVector}

/**
  * Solves the optimization problem pertaining
  * to the weights of a committee model.
  */
class CommitteeModelSolver extends
RegularizedOptimizer[DenseVector[Double],
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
