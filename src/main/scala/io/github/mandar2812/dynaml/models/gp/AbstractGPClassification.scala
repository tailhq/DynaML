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

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.models.ParameterizedLearner

/**
  * @author mandar on 6/4/16.
  *
  * Skeleton of a Gaussian Process binary classification
  * model.
  *
  * @tparam T The type of data structure holding the training instances
  *
  * @tparam I The type of input features (also called index set)
  *
  * @tparam P The data type storing all the relevant information
  *           about the posterior mode of the nuisance function.
  *
  */
abstract class AbstractGPClassification[T, I, P](data: T) extends
  GaussianProcessModel[T, I, Double, Double, DenseMatrix[Double],
  (DenseVector[Double], DenseMatrix[Double])] with
  ParameterizedLearner[T, P, I, Double,
    (DenseMatrix[Double], DenseVector[Double])] {


  override protected val g: T = data

  val num_points = dataAsSeq(g).length

  override protected var params: P = initParams()

  /**
    * Learn the Laplace approximation
    * to the posterior mean of the nuisance
    * function f(x) ~ GP(m(x), K(x, x'))
    *
    **/
  override def learn(): Unit = {
    val procdata = dataAsIndexSeq(g)
    val targets = DenseVector(dataAsSeq(g).map(_._2).toArray)
    val kernelMat = covariance.buildKernelMatrix(procdata, procdata.length).getKernelMatrix()
    params = optimizer.optimize(
      num_points,
      (kernelMat, targets),
      initParams()
    )
  }


}
