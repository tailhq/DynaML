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
import io.github.mandar2812.dynaml.kernels.LocalScalarKernel
import io.github.mandar2812.dynaml.pipes.DataPipe

/**
  * @author mandar2812
  *
  * GP-NARX
  * Gaussian Process Non-Linear
  * Auto-regressive Model with
  * Exogenous Inputs.
  *
  * y(t) = f(x(t)) + e
  * x(t) = (y(t-1), ... , y(t-p), u(t-1), ..., u(t-p))
  * f(x) ~ GP(0, cov(X,X))
  * e|f(x) ~ N(0, noise(X,X))
  */
class GPNarXModel(order: Int,
                  ex: Int,
                  cov: LocalScalarKernel[DenseVector[Double]],
                  nL: LocalScalarKernel[DenseVector[Double]],
                  trainingdata: Seq[(DenseVector[Double], Double)],
                  meanFunc: DataPipe[DenseVector[Double], Double] = DataPipe(_ => 0.0)) extends
GPRegression(cov, nL, trainingdata, meanFunc) {

  val modelOrder = order

  val exogenousInputs = ex

}