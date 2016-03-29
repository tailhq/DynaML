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

package io.github.mandar2812.dynaml.models.lm

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.models.LinearModel
import io.github.mandar2812.dynaml.optimization.{RegularizedLSSolver, RegularizedOptimizer}

/**
  * Created by mandar on 29/3/16.
  */
class RegularizedGLM(data: Stream[(DenseVector[Double], Double)],
                     numPoints: Int,
                     map: (DenseVector[Double]) => DenseVector[Double] =
                     identity[DenseVector[Double]] _)
  extends LinearModel[Stream[(DenseVector[Double], Double)],
    Int, Int, DenseVector[Double], DenseVector[Double], Double,
    (DenseMatrix[Double], DenseVector[Double])]{

  override protected val g = data

  def dimensions = featureMap(data.head._1).length

  override def initParams(): DenseVector[Double] =
    DenseVector.ones[Double](dimensions)


  override protected val optimizer: RegularizedOptimizer[Int, DenseVector[Double],
    DenseVector[Double], Double,
    (DenseMatrix[Double], DenseVector[Double])] = new RegularizedLSSolver

  override protected var params: DenseVector[Double] = initParams()

  override def clearParameters(): Unit = {
    params = initParams()
  }


  /**
    * Learn the parameters
    * of the model which
    * are in a node of the
    * graph.
    *
    **/
  override def learn(): Unit = {
    val designMatrix = DenseMatrix.vertcat[Double](
      g.map(point => featureMap(point._1).toDenseMatrix):_*
    )

    val responseVector = DenseVector.vertcat(
      g.map(p => DenseVector(p._2)):_*
    )

    params = optimizer.optimize(numPoints,
      (designMatrix, responseVector),
      initParams())
  }

  /**
    * Predict the value of the
    * target variable given a
    * point.
    *
    **/
  override def predict(point: DenseVector[Double]): Double =
    params dot featureMap(point)

}
