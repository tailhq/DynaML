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
package io.github.mandar2812.dynaml.models.svm

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.kernels.CovarianceFunction
import io.github.mandar2812.dynaml.models.LinearModel
import io.github.mandar2812.dynaml.optimization.GloballyOptimizable

/**
  * Created by mandar on 3/4/16.
  */
abstract class AbstractDualLSSVM[Index](data: Stream[(Index, Double)],
                                        numPoints: Int,
                                        kern: CovarianceFunction[Index, Double,
                                          DenseMatrix[Double]])
  extends LinearModel[Stream[(Index, Double)], DenseVector[Double],
    Index, Double, (DenseMatrix[Double], DenseVector[Double])]
    with GloballyOptimizable {

  override protected val g = data

  val kernel = kern

  override protected var current_state: Map[String, Double] =
    Map("regularization" -> 0.1) ++ kernel.state

  override protected var hyper_parameters: List[String] =
    List("regularization")++kernel.hyper_parameters

  val num_points = numPoints

  /**
    * Initialize the synapse weights
    * to small random values between
    * 0 and 1.
    *
    * */
  override def initParams(): DenseVector[Double] =
    DenseVector.ones[Double](num_points+1)

  override def clearParameters(): Unit = {
    params = initParams()
  }

  override protected var params: DenseVector[Double] = initParams()


  /**
    * Predict the value of the
    * target variable given a
    * point.
    *
    **/
  override def predict(point: Index): Double = {

    val features = DenseVector(g.map(inducingpoint =>
      kernel.evaluate(point, inducingpoint._1)).toArray)

    params(0 until num_points) dot features + params(-1)
  }

  def setState(h: Map[String, Double]) = {
    kernel.setHyperParameters(h)
    current_state += ("regularization" -> h("regularization"))
  }


}
