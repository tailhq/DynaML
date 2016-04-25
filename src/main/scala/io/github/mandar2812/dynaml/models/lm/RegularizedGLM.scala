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
import io.github.mandar2812.dynaml.models.gp.AbstractGPRegressionModel
import io.github.mandar2812.dynaml.optimization.{GloballyOptimizable, RegularizedLSSolver, RegularizedOptimizer}

/**
  * Created by mandar on 29/3/16.
  */
class RegularizedGLM(data: Stream[(DenseVector[Double], Double)],
                     numPoints: Int,
                     map: (DenseVector[Double]) => DenseVector[Double] =
                     identity[DenseVector[Double]] _)
  extends GeneralizedLinearModel[(DenseMatrix[Double], DenseVector[Double])](data, numPoints, map)
    with GloballyOptimizable {

  override val task = "regression"

  override protected val optimizer: RegularizedOptimizer[DenseVector[Double],
    DenseVector[Double], Double,
    (DenseMatrix[Double], DenseVector[Double])] = new RegularizedLSSolver


  override def prepareData(d: Stream[(DenseVector[Double], Double)]) = {
    val designMatrix = DenseMatrix.vertcat[Double](
      d.map(point => DenseVector(featureMap(point._1).toArray ++ Array(1.0)).toDenseMatrix):_*
    )

    val responseVector = DenseVector.vertcat(
      d.map(p => DenseVector(p._2)):_*
    )

    (designMatrix, responseVector)
  }
}
