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

import breeze.linalg.{DenseMatrix, DenseVector, inv}
import io.github.mandar2812.dynaml.pipes.DataPipe
import org.apache.log4j.Logger
import spire.implicits._

/**
  * @author mandar2812 date: 16/4/16.
  *
  * An abstract base class for Quasi Newton based
  * convex optimization methods.
  *
  * It requires two components
  * 1. A gradient implementation
  * 2. An updater which computes the approximate
  * inverse Hessian and carries out the update.
  */
class QuasiNewtonOptimizer(private var gradient: Gradient,
                           private var updater: HessianUpdater)
  extends RegularizedOptimizer[DenseVector[Double],
    DenseVector[Double], Double,
    Stream[(DenseVector[Double], Double)]]{

  private val logger = Logger.getLogger(this.getClass)

  /**
    * Solve the convex optimization problem.
    */
  override def optimize(nPoints: Long,
                        ParamOutEdges: Stream[(DenseVector[Double], Double)],
                        initialP: DenseVector[Double]): DenseVector[Double] =
    QuasiNewtonOptimizer.run(
      nPoints, this.regParam, this.numIterations,
      updater, gradient, this.stepSize, initialP,
      ParamOutEdges, DataPipe(identity[Stream[(DenseVector[Double], Double)]] _)
    )
}

object QuasiNewtonOptimizer {

  private val logger = Logger.getLogger(this.getClass)

  def run[T](nPoints: Long, regParam: Double, numIterations: Int,
             updater: HessianUpdater, gradient: Gradient, stepSize: Double,
             initial: DenseVector[Double], POutEdges: T,
             transform: DataPipe[T, Stream[(DenseVector[Double], Double)]]): DenseVector[Double] = {

    var oldW: DenseVector[Double] = initial

    var newW = oldW
    val hessian = transform(POutEdges)
      .map(_._1)
      .map(x => DenseVector(x.toArray ++ Array(1.0)))
      .map(x => x*x.t)
      .reduce((x: DenseMatrix[Double],
               y: DenseMatrix[Double]) =>
        x + y)

    var regInvHessian = inv(hessian + DenseMatrix.eye[Double](initial.length)*regParam)
    var oldCumGradient = DenseVector.zeros[Double](initial.length)

    cfor(1)(iter => iter < numIterations, iter => iter + 1)( iter => {
      val cumGradient: DenseVector[Double] = DenseVector.zeros(initial.length)
      var cumLoss: Double = 0.0
      transform(POutEdges).foreach(ed => {
        val x = DenseVector(ed._1.toArray ++ Array(1.0))
        val y = ed._2
        cumLoss += gradient.compute(x, y, oldW, cumGradient)
      })

      logger.info("Average Loss; Iteration "+iter+": "+cumLoss/nPoints.toDouble)
      //Find the search direction p = inv(H)*grad(J)
      //perform update x_new = x + step*p
      val searchDirection = regInvHessian*cumGradient*(-1.0)
      newW = updater.compute(oldW, searchDirection, stepSize, iter, regParam)._1

      regInvHessian = updater.hessianUpdate(regInvHessian, newW-oldW, cumGradient-oldCumGradient)
      oldW = newW
      oldCumGradient = cumGradient
    })
    newW
  }
}
