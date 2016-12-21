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

import breeze.linalg.{DenseMatrix, DenseVector, cholesky, inv}
import breeze.numerics.sqrt
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.probability.Likelihood

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
                        initialP: DenseVector[Double]): DenseVector[Double] =
    LaplacePosteriorMode.run(
      nPoints, ParamOutEdges,
      this.likelihood, initialP,
      this.numIterations, identityPipe[(DenseMatrix[Double], DenseVector[Double])])
}

object LaplacePosteriorMode {

  def run[T](nPoints: Long, data: T,
             likelihood: Likelihood[
               DenseVector[Double], DenseVector[Double], DenseMatrix[Double],
               (DenseVector[Double], DenseVector[Double])],
             initialP: DenseVector[Double], numIterations: Int,
             transform: DataPipe[T, (DenseMatrix[Double], DenseVector[Double])]): DenseVector[Double] = {

    val (kMat, y) = transform(data)
    var mode = initialP

    var b = DenseVector.zeros[Double](y.length)
    var a = DenseVector.zeros[Double](y.length)

    val id = DenseMatrix.eye[Double](y.length)

    (1 to numIterations).foreach{ iter =>
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
