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
package io.github.mandar2812.dynaml.models.gp

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.models.LinearModel
import io.github.mandar2812.dynaml.pipes.GPRegressionPipe

/**
  * Created by mandar on 9/2/16.
  */
abstract class GPCommitteeRegression[T,D[T]](num: Int, data: D[T],
                                             networks: GPRegressionPipe[GPRegression, D[T]]*)
  extends LinearModel[D[T], DenseVector[Double], DenseVector[Double], Double, D[T]] {

  override protected val g: D[T] = data

  val baseNetworks: List[GPRegression] =
    networks.toList.map(net => net.run(g))

  val num_points = num

  /**
    * Predict the value of the
    * target variable given a
    * point.
    *
    **/
  override def predict(point: DenseVector[Double]): Double =
    params dot featureMap(point)

  override def clearParameters(): Unit =
    DenseVector.fill[Double](baseNetworks.length)(1.0)

  override def initParams(): DenseVector[Double] =
    DenseVector.fill[Double](baseNetworks.length)(1.0)

  /**
    * Learn the parameters
    * of the model which
    * are in a node of the
    * graph.
    *
    **/
  override def learn(): Unit = {

    params = optimizer.optimize(
      num_points,
      g, initParams())
  }

  override protected var params: DenseVector[Double] =
    DenseVector.fill[Double](baseNetworks.length)(1.0)

  featureMap = (pattern) =>
    DenseVector(baseNetworks.map(net =>
      net.predictionWithErrorBars(Seq(pattern), 1).head._2
    ).toArray)

}
