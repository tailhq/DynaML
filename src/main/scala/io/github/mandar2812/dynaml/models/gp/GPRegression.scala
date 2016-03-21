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
import io.github.mandar2812.dynaml.kernels.{DiracKernel, CovarianceFunction}

/**
  *
  * @author mandar2812 date 17/11/15.
  *
  * Class representing Gaussian Process regression models
  *
  * y = f(x) + e
  * f(x) ~ GP(0, cov(X,X))
  * e|f(x) ~ N(f(x), noise(X,X))
  */
class GPRegression(
  cov: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]],
  noise: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]] = new DiracKernel(1.0),
  trainingdata: Seq[(DenseVector[Double], Double)]) extends
AbstractGPRegressionModel[Seq[(DenseVector[Double], Double)],
  DenseVector[Double]](cov, noise, trainingdata,
  trainingdata.length){

  /**
    * Convert from the underlying data structure to
    * Seq[(I, Y)] where I is the index set of the GP
    * and Y is the value/label type.
    **/
  override def dataAsSeq(data: Seq[(DenseVector[Double], Double)]) = data
}
